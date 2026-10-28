import torch
import torch.nn as nn


class HybridLoss(nn.Module):
    '''returns "total","cplx","mag","sisnr"'''
    def __init__(self):
        super().__init__()

    def forward(self, pred_stft, true_stft):
        device = pred_stft.device

        pred_stft_real, pred_stft_imag = torch.real(pred_stft), torch.imag(pred_stft)
        true_stft_real, true_stft_imag = torch.real(true_stft), torch.imag(true_stft)
        pred_mag = torch.sqrt(pred_stft_real**2 + pred_stft_imag**2 + 1e-12)
        true_mag = torch.sqrt(true_stft_real**2 + true_stft_imag**2 + 1e-12)
        pred_real_c = pred_stft_real / (pred_mag**(0.7))
        pred_imag_c = pred_stft_imag / (pred_mag**(0.7))
        true_real_c = true_stft_real / (true_mag**(0.7))
        true_imag_c = true_stft_imag / (true_mag**(0.7))
        real_loss = nn.MSELoss()(pred_real_c, true_real_c)
        imag_loss = nn.MSELoss()(pred_imag_c, true_imag_c)
        mag_loss = nn.MSELoss()(pred_mag**(0.3), true_mag**(0.3))
        
        y_pred = torch.istft(pred_stft.squeeze(), 512, 256, 512, window=torch.hann_window(512).pow(0.5).to(device))
        y_true = torch.istft(true_stft.squeeze(), 512, 256, 512, window=torch.hann_window(512).pow(0.5).to(device))
        y_true = torch.sum(y_true * y_pred, dim=-1, keepdim=True) * y_true / (torch.sum(torch.square(y_true),dim=-1,keepdim=True) + 1e-8)

        sisnr =  - 10 * torch.log10(torch.norm(y_true, dim=-1, keepdim=True)**2 / (torch.norm(y_pred - y_true, dim=-1, keepdim=True)**2+1e-8) + 1e-8).mean()

        #return 30*(real_loss + imag_loss) + 70*mag_loss + sisnr
        total = 30 * (real_loss + imag_loss) + 70 * mag_loss + sisnr

        return {
            "total": total,
            "cplx": real_loss.detach()+imag_loss.detach(),
            "mag":  mag_loss.detach(),
            "sisnr": sisnr.detach(),
        }

def sisnr_metric(ref, inf):
    import numpy as np
    inf = inf - inf.mean()
    ref = ref - ref.mean()

    a = np.sum(inf * ref) / np.sum(ref**2 + 1e-8)
    e_tagt = a * ref
    e_res = inf - e_tagt

    return 10*np.log10((np.sum(e_tagt**2)+1e-8) / (np.sum(e_res**2)+1e-8))

if __name__ == "__main__":
    loss_func = HybridLoss()
    import soundfile as sf
    noisy = sf.read("/data/ssd0/jizhe.lu/Simulated/TEST/sample_0_clean_fileid_9027_test.wav", dtype="float32")[0]
    print("noisy", noisy.shape)
    clean = sf.read("/data/ssd0/jizhe.lu/Simulated/TEST_TRUTH/RTS/test_clean_fileid_9027_RTS.wav", dtype="float32")[0]
    noisy_stft = torch.stft(torch.from_numpy(noisy[:,0]).float(), 512, 256, 512, window=torch.hann_window(512).pow(0.5), return_complex=True).unsqueeze(0)
    clean_stft = torch.stft(torch.from_numpy(clean).float(), 512, 256, 512, window=torch.hann_window(512).pow(0.5), return_complex=True).unsqueeze(0)
    
    
    loss = loss_func(noisy_stft, clean_stft)
    sisnr = sisnr_metric(clean, noisy[:,0])
    print(loss)
    print(sisnr)
