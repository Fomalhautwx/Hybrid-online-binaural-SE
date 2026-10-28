"""从双掩码 ckpt 导出流式 StreamGTCRNDual 到 ONNX（含 mask2_gate 烘焙）。

用法：python export_onnx_dual.py <ckpt.tar> <out.onnx>
"""
import os
import sys
import torch
import onnx
from onnxsim import simplify

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.gtcrn_paris_stream import StreamGTCRNDual


def load_stream_from_sd(stream_model, src_sd):
    new_sd = stream_model.state_dict()
    for key in list(new_sd.keys()):
        if key in src_sd:
            if ('point_conv' in key) and 'decode' in key and key.endswith('weight'):
                new_sd[key] = src_sd[key].permute([1, 0, 2, 3])
            else:
                new_sd[key] = src_sd[key]
        elif key.replace('Conv1d.', '') in src_sd:
            new_sd[key] = src_sd[key.replace('Conv1d.', '')]
        elif key.replace('Conv2d.', '') in src_sd:
            new_sd[key] = src_sd[key.replace('Conv2d.', '')]
        elif key.replace('ConvTranspose2d.', '') in src_sd:
            k2 = key.replace('ConvTranspose2d.', '')
            if key.endswith('weight'):
                new_sd[key] = src_sd[k2] if new_sd[key].shape == src_sd[k2].shape else src_sd[k2].permute([1, 0, 2, 3])
            else:
                new_sd[key] = src_sd[k2]
        else:
            raise ValueError(f'key 无法映射: {key}')
    stream_model.load_state_dict(new_sd, strict=True)


def main():
    ckpt = sys.argv[1]
    out = sys.argv[2]
    device = torch.device('cpu')

    stream_model = StreamGTCRNDual().to(device).eval()
    sd = torch.load(ckpt, map_location=device)
    load_stream_from_sd(stream_model, sd['model'])
    print(f"已载入 {ckpt}，mask2_gate = {stream_model.mask2_gate.item():.4f}")

    input = torch.randn(1, 2, 257, 1, 2, device=device)
    ref_dummy = torch.randn(1, 2, 257, 1, 2, device=device)
    conv_cache = torch.zeros(2, 1, 32, 16, 65, device=device)
    tra_cache = torch.zeros(2, 3, 1, 1, 32, device=device)
    inter_cache = torch.zeros(2, 1, 65, 32, device=device)

    torch.onnx.export(
        stream_model, (input, ref_dummy, conv_cache, tra_cache, inter_cache), out,
        input_names=['spec', 'ref', 'conv_cache', 'tra_cache', 'inter_cache'],
        output_names=['spec_enh', 'conv_cache_out', 'tra_cache_out', 'inter_cache_out'],
        opset_version=11)

    model_simp, check = simplify(onnx.load(out))
    assert check, "ONNX simplify 校验失败"
    onnx.save(model_simp, out.replace('.onnx', '_simple.onnx'))
    print(f"已导出 {out} 与 {out.replace('.onnx', '_simple.onnx')}")


if __name__ == '__main__':
    main()
