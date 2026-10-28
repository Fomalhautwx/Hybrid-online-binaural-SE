import yaml

from utils.iva_debug import NaNProbe
from models.AuxIVA_ISS_harmonic_v2 import AuxIVA_ISS_harmonic_nn_Frame_Module
import torch
from models.Paris_enhancement import GTCRN_PARIS

# d = torch.load("nan_dumps/unknown_b5_t155.pt")
# iva = AuxIVA_ISS_harmonic_nn_Frame_Module(alpha2=0)
# NaNProbe(iva).run(d["X"], d["P"]).print()

"""
把这段贴进你的环境跑。改三处：model的构建、batch的加载路径、前向拆分方式。
"""

# ---- 1. 加载坏 batch ----
d = torch.load("/home/nis/jizhe.lu/Code/DNS3_2Mix_noisy_reverb/train_dump/debug111.pt")  # 改成你的路径


# 如果 guard 没有 dump（比如用的是旧版），就手动拿：
# 在训练循环里第 42 步前加 torch.save(batch, "batch42.pt")
# batch = torch.load("batch42.pt")

# ---- 2. 检查输入本身 --

# ---- 3. 逐段前向 ----
# 把你的 model.forward() 拆成几段，每段之后查一次。
# 下面是一个典型的 encoder → IVA → decoder 结构的示例，你按自己的改。
with open("config.yaml", 'r') as file:
        config = yaml.safe_load(file)
model = GTCRN_PARIS(**config["network"], params=config["network"])  
model.train()
model.cuda()  # 如果用 GPU

with torch.no_grad():
    x = d["X"].cuda()        # 改成你的键名
    ref = d["prior"].cuda()    # 改成你的键名

    # ---- 编码器 ----
    est, dummy_out = model(x, ref)
    print(f"max={est[torch.isfinite(est)].abs().max():.3e}")