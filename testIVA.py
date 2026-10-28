import os
import soundfile as sf
from models.AuxIVA import AuxIVA_numpy, auxiva, batch_auxiva, multi_channel_stft, multi_channel_istft
from models.AuxIVA_ICD_RLS_erb import AuxIVA_ICD_RLS_erb_nn_v2_Module
from models.AuxIVA_ICD_RLS_harmonic import AuxIVA_ICD_RLS_harmonic_nn_v2_Module
from models.AuxIVA_ISS_harmonic import AuxIVA_ISS_harmonic_nn_Frame_Module
import torch


def test_auxiva_iterwise(
    wav_path,
    out_dir,
    nn_prior_path,
    nfft=512,
    hop=256,
    max_iter=5,
):
    """
    wav_path: 双通道混合语音 wav
    out_dir:  输出目录
    """

    os.makedirs(out_dir, exist_ok=True)
    auxiva_icd = AuxIVA_ICD_RLS_harmonic_nn_v2_Module(alpha2=0.2)  # instantiate iva
    auxiva_iss = AuxIVA_ISS_harmonic_nn_Frame_Module(alpha2=0)  # instantiate iva
    
    
    # ---- load stereo wav ----
    x, sr = sf.read(wav_path)  # (T, 2)
    assert x.ndim == 2 and x.shape[1] == 2, "Input must be stereo"
    clean, _ = sf.read(nn_prior_path)  # (T,)
    clean = torch.from_numpy(clean).float()  # (T,)
    # (nch, T)
    #x = x.T
    x = torch.from_numpy(x).float()
    print(f"Loaded wav: {wav_path}, sr={sr}, shape={x.shape}")


    # ---- run AuxIVA with increasing n_iter ----
    for n_iter in [1]:
        print(f"\n=== Running AuxIVA: n_iter={n_iter} ===")

        '''y_offline_auxiva = AuxIVA_numpy(
            x.T,
            nfft=nfft,
            nol=hop,
            n_iter=n_iter,
            Lb=16,
            verbose=False
        )'''
        X_batch = multi_channel_stft(x.unsqueeze(0), n_fft=512, hop_length=256, win_length=512, window=torch.hann_window(512).pow(0.5))  # (1, nch, n_freq, n_frames)
        print(clean.shape)
        CLean = multi_channel_stft(clean.unsqueeze(1).unsqueeze(0), n_fft=512, hop_length=256, win_length=512, window=torch.hann_window(512).pow(0.5))  # (1, 1, n_freq, n_frames)
        
        # y = batch_auxiva(
        #     X_batch,
        #     n_iter=n_iter
        # )
        y = auxiva_icd(
            X_batch,
            CLean
        )  # (1, nsrc, n_freq, n_frames)
        # y: (nsrc, T)
        y = multi_channel_istft(y, n_fft=512, hop_length=256, win_length=512, window=torch.hann_window(512).pow(0.5))  # (nsrc, T)
        
        y = y.detach().cpu().numpy()
        # ---- save each source ----
        out_path = os.path.join(
            out_dir,
            f"auxiva_icd_harmonic_offline.wav"
        )
        
        print(y.shape)
        sf.write(out_path, y.squeeze(0).T, sr)
        #interf = multi_channel_istft(interf, n_fft=512, hop_length=256, win_length=512, window=torch.hann_window(512).pow(0.5))  # (nsrc, T)
        #sf.write(os.path.join(out_dir, "residual.wav"), interf.squeeze().T, sr)
        print(f"Saved: {out_path}")
        
        # stream_path = '/home/nis/jizhe.lu/Code/DNS3_2Mix_noisy_reverb/test_wavs/enh_stream.wav'
    import numpy as np
    from metrics import pesq_scorer, stoi_scorer
    import fast_bss_eval
    clean_path = 'clean.wav'
    clean = sf.read(clean_path)[0]
    # stream = sf.read(stream_path)[0]
    clean = np.expand_dims(clean, axis=0)
    noisy = x.detach().cpu().numpy().T[0:1]
    print(noisy.shape)
    
    auxiva = sf.read("output/tcsun_iva/auxiva_offline_iter1.wav")[0].T[1:2]
    icd_erb = sf.read("output/tcsun_iva/auxiva_icd_erb_offline.wav")[0].T[1:2]
    icd_harmonic = sf.read("output/tcsun_iva/auxiva_icd_harmonic_offline.wav")[0].T[0:1]
    print(auxiva.shape, icd_erb.shape, icd_harmonic.shape)
    
    
    mix_sisdr = fast_bss_eval.si_sdr(clean, noisy)
    auxiva_sisdr = fast_bss_eval.si_sdr(clean, auxiva)
    erb_sisdr = fast_bss_eval.si_sdr(clean, icd_erb)
    harmonic_sisdr = fast_bss_eval.si_sdr(clean, icd_harmonic)
    stoi = stoi_scorer()
    pesq = pesq_scorer()
    print('STOI')
    print(f"auxiva {stoi(noisy, clean)}->{stoi(auxiva, clean)}")
    print(f"erb {stoi(noisy, clean)}->{stoi(icd_erb, clean)}")
    print(f"harmonic {stoi(noisy, clean)}->{stoi(icd_harmonic, clean)}")
    print('PESQ')
    print(f"auxiva {pesq(noisy, clean)}->{pesq(auxiva, clean)}")
    print(f"erb {pesq(noisy, clean)}->{pesq(icd_erb, clean)}")
    print(f"harmonic {pesq(noisy, clean)}->{pesq(icd_harmonic, clean)}")
    print('Sisdr')
    print(f"auxiva {mix_sisdr}->{auxiva_sisdr}")
    print(f"erb {mix_sisdr}->{erb_sisdr}")
    print(f"harmonic {mix_sisdr}->{harmonic_sisdr}")

if __name__ == "__main__":
    # Example usage
    test_auxiva_iterwise(
        wav_path="/home/nis/jizhe.lu/Code/2sderev/results/E2EpWPE_HA/offline.wav",
        #wav_path="/home/nis/jizhe.lu/Code/DNS3_2Mix_noisy_reverb/sample/inspect/enh_der_train.wav",
        out_dir="output/tcsun_iva",
        nn_prior_path="clean.wav",
        nfft=512,
        hop=256,
        max_iter=1,
    )