import csv
import torch
import numpy as np
import logging
import torch.nn as nn
from pesq import pesq
from pystoi import stoi
from joblib import Parallel, delayed

import fast_bss_eval
from loss_modules.paired_loss import PairedNegSDR
from loss_modules.pit_loss import PITLossWrapper
from loss_modules.pairwise_loss import PairwiseNegSDR


logger = logging.getLogger(__name__)


class MetricsTracker:
    def __init__(self, save_file: str = ""):
        self.all_sdrs = []
        self.all_sdrs_i = []
        self.all_sisnrs = []
        self.all_sisnrs_i = []
        csv_columns = ["snt_id", "sdr", "sdr_i", "si-snr", "si-snr_i"]
        self.results_csv = open(save_file, "w")
        self.writer = csv.DictWriter(self.results_csv, fieldnames=csv_columns)
        self.writer.writeheader()
        self.pit_sisnr = PITLossWrapper(
            PairwiseNegSDR("sisdr", zero_mean=False), pit_from="pw_mtx"
        )
        self.pit_snr = PITLossWrapper(
            PairwiseNegSDR("snr", zero_mean=False), pit_from="pw_mtx"
        )

    def __call__(self, mix, clean, estimate, key):
        # sisnr
        sisnr = self.pit_sisnr(estimate.unsqueeze(0), clean.unsqueeze(0))
        mix = torch.stack([mix] * clean.shape[0], dim=0)
        sisnr_baseline = self.pit_sisnr(mix.unsqueeze(0), clean.unsqueeze(0))
        sisnr_i = sisnr - sisnr_baseline

        # sdr
        sdr = -fast_bss_eval.sdr_pit_loss(estimate, clean).mean()
        sdr_baseline = -fast_bss_eval.sdr_pit_loss(mix, clean).mean()
        sdr_i = sdr - sdr_baseline
        # import pdb; pdb.set_trace()
        row = {
            "snt_id": key,
            "sdr": sdr.item(),
            "sdr_i": sdr_i.item(),
            "si-snr": -sisnr.item(),
            "si-snr_i": -sisnr_i.item(),
        }
        self.writer.writerow(row)
        # Metric Accumulation
        self.all_sdrs.append(sdr.item())
        self.all_sdrs_i.append(sdr_i.item())
        self.all_sisnrs.append(-sisnr.item())
        self.all_sisnrs_i.append(-sisnr_i.item())
    
    def update(self, ):
        return {"sdr_i": np.array(self.all_sdrs_i).mean(),
                "si-snr_i": np.array(self.all_sisnrs_i).mean()
                }

    def final(self,):
        row = {
            "snt_id": "avg",
            "sdr": np.array(self.all_sdrs).mean(),
            "sdr_i": np.array(self.all_sdrs_i).mean(),
            "si-snr": np.array(self.all_sisnrs).mean(),
            "si-snr_i": np.array(self.all_sisnrs_i).mean(),
        }
        self.writer.writerow(row)
        row = {
            "snt_id": "std",
            "sdr": np.array(self.all_sdrs).std(),
            "sdr_i": np.array(self.all_sdrs_i).std(),
            "si-snr": np.array(self.all_sisnrs).std(),
            "si-snr_i": np.array(self.all_sisnrs_i).std(),
        }
        self.writer.writerow(row)
        self.results_csv.close()

class pesq_scorer(nn.Module):
    def __init__(self, fs=16000):
        super().__init__()
        self.fs = fs

    def forward(self, pred_wav, true_wav):
        """
            pred_wav: (B, C, L) or (B, L)
            true_wav: (B, C, L) or (B, L)
        """
        device = pred_wav.device if torch.is_tensor(pred_wav) else torch.device('cpu')
        
        if torch.is_tensor(pred_wav):
            pred_wav = pred_wav.detach().cpu().numpy()
        if torch.is_tensor(true_wav):
            true_wav = true_wav.detach().cpu().numpy()
        
        if pred_wav.ndim == 3:
            pred_wav = pred_wav.reshape(-1, pred_wav.shape[-1])
        
        if true_wav.ndim == 3:
            true_wav = true_wav.reshape(-1, true_wav.shape[-1])
        
        data_len = min(pred_wav.shape[-1], true_wav.shape[-1])
        pred_wav = pred_wav[..., : data_len]
        true_wav = true_wav[..., : data_len]
        
        pesq_score_batch = Parallel(n_jobs=4, backend="threading")( 
            delayed(pesq)(self.fs, _true, _pred) for _true, _pred in zip(true_wav, pred_wav))
        pesq_score = torch.tensor(pesq_score_batch, device=device).mean()

        return pesq_score
    

class stoi_scorer(nn.Module):
    def __init__(self, fs=16000, extended=False):
        super().__init__()
        
        self.fs = fs
        self.extended = extended
        
    def forward(self, pred_wav, true_wav):
        """
            pred_wav: (B, C, L) or (B, L)
            true_wav: (B, C, L) or (B, L)
        """
        device = pred_wav.device if torch.is_tensor(pred_wav) else torch.device('cpu')
        
        if torch.is_tensor(pred_wav):
            pred_wav = pred_wav.detach().cpu().numpy()
        if torch.is_tensor(true_wav):
            true_wav = true_wav.detach().cpu().numpy()
        
        if pred_wav.ndim == 3:
            pred_wav = pred_wav.reshape(-1, pred_wav.shape[-1])
        
        if true_wav.ndim == 3:
            true_wav = true_wav.reshape(-1, true_wav.shape[-1])
        
        data_len = min(pred_wav.shape[-1], true_wav.shape[-1])
        pred_wav = pred_wav[..., : data_len]
        true_wav = true_wav[..., : data_len]
        
        stoi_score_batch = Parallel(n_jobs=4, backend="threading")( 
            delayed(stoi)(_true, _pred, self.fs, self.extended) for _true, _pred in zip(true_wav, pred_wav))
        stoi_score = torch.tensor(stoi_score_batch, device=device).mean()

        return stoi_score
    
    
def pad_x_to_y(x, y, axis: int = -1):
    if axis != -1:
        raise NotImplementedError
    inp_len = y.shape[axis]
    output_len = x.shape[axis]
    return nn.functional.pad(x, [0, inp_len - output_len])


def shape_reconstructed(reconstructed, size):
    if len(size) == 1:
        return reconstructed.squeeze(0)
    return reconstructed


def tensors_to_device(tensors, device):
    """Transfer tensor, dict or list of tensors to device.

    Args:
        tensors (:class:`torch.Tensor`): May be a single, a list or a
            dictionary of tensors.
        device (:class: `torch.device`): the device where to place the tensors.

    Returns:
        Union [:class:`torch.Tensor`, list, tuple, dict]:
            Same as input but transferred to device.
            Goes through lists and dicts and transfers the torch.Tensor to
            device. Leaves the rest untouched.
    """
    if isinstance(tensors, torch.Tensor):
        return tensors.to(device)
    elif isinstance(tensors, (list, tuple)):
        return [tensors_to_device(tens, device) for tens in tensors]
    elif isinstance(tensors, dict):
        for key in tensors.keys():
            tensors[key] = tensors_to_device(tensors[key], device)
        return tensors
    else:
        return tensors
    