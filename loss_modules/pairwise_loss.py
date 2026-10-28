import torch
import torch.nn as nn


class PairwiseNegSDR(nn.Module):
    r"""
        Base class for pairwise negative SI-SDR, SD-SDR, SNR, t-SDR, eps-t-SDR on a batch.

        input:
            est_targets: (B, nsrc, L)
            targets: (B, nsrc, L)
        
        return:
            (batch, n_src, n_src)
    """

    def __init__(self, sdr_type, zero_mean=True, take_log=True, EPS=1e-8, SDR_max=30):
        super(PairwiseNegSDR, self).__init__()
        assert sdr_type in ["snr", "sisdr", "sdsdr", "tsdr", "eps_tsdr"]
        self.sdr_type = sdr_type
        self.zero_mean = zero_mean
        self.take_log = take_log
        self.EPS = EPS
        self.tau = 10 ** (-SDR_max / 10)
        
    def calculate_sdr(self, s_estimate, s_target):
        if self.sdr_type == "tsdr":
            # [batch, n_src, n_src, time]
            pair_wise_proj = s_target.repeat(1, s_target.shape[2], 1, 1)
            
            e_noise = s_estimate - s_target
            
            # [batch, n_src, n_src]
            pair_wise_sdr = torch.sum(pair_wise_proj**2, dim=3) / (
                torch.sum(e_noise**2, dim=3) + self.tau * torch.sum(pair_wise_proj**2, dim=3)
            )
            
            return pair_wise_sdr
        
        if self.sdr_type == "eps_tsdr":
            # [batch, n_src, n_src, time]
            pair_wise_proj = s_target.repeat(1, s_target.shape[2], 1, 1)
            
            e_noise = s_estimate - s_target
            
            # [batch, n_src, n_src]            
            pair_wise_sdr = ( torch.sum(pair_wise_proj**2, dim=3) + self.EPS) / (
                torch.sum(e_noise**2, dim=3) + self.tau * (torch.sum(pair_wise_proj**2, dim=3) + self.EPS)
            )
            
            return pair_wise_sdr
        
        # sisdr, sdsdr, snr
        if self.sdr_type in ["sisdr", "sdsdr"]:
            # [batch, n_src, n_src, 1]
            pair_wise_dot = torch.sum(s_estimate * s_target, dim=3, keepdim=True)
            # [batch, 1, n_src, 1]
            s_target_energy = torch.sum(s_target**2, dim=3, keepdim=True) + self.EPS
            # [batch, n_src, n_src, time]
            pair_wise_proj = pair_wise_dot * s_target / s_target_energy
        else:
            # [batch, n_src, n_src, time]
            pair_wise_proj = s_target.repeat(1, s_target.shape[2], 1, 1)
            
        if self.sdr_type in ["sdsdr", "snr"]:
            e_noise = s_estimate - s_target
        else:
            e_noise = s_estimate - pair_wise_proj
            
        # [batch, n_src, n_src]
        pair_wise_sdr = torch.sum(pair_wise_proj**2, dim=3) / (
            torch.sum(e_noise**2, dim=3) + self.EPS
        )
        
        return pair_wise_sdr
        

    def forward(self, est_targets, targets):
        if targets.size() != est_targets.size() or targets.ndim != 3:
            raise TypeError(
                f"Inputs must be of shape [batch, n_src, time], got {targets.size()} and {est_targets.size()} instead"
            )
        assert targets.size() == est_targets.size()
        
        # Step 1. Zero-mean norm
        if self.zero_mean:
            mean_source = torch.mean(targets, dim=2, keepdim=True)
            mean_estimate = torch.mean(est_targets, dim=2, keepdim=True)
            targets = targets - mean_source
            est_targets = est_targets - mean_estimate
            
        # Step 2. Pair-wise PairedNegSDR. (Reshape to use broadcast)
        # [batch, n_src, 1, time]
        s_estimate = torch.unsqueeze(est_targets, dim=2)
        # [batch, 1, n_src, time]
        s_target = torch.unsqueeze(targets, dim=1)
            
        pair_wise_sdr = self.calculate_sdr(s_estimate, s_target)
        
        if self.take_log:
            pair_wise_sdr = 10 * torch.log10(pair_wise_sdr + self.EPS)
        return -pair_wise_sdr