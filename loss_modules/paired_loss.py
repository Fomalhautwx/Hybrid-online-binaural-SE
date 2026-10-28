import torch
import torch.nn as nn


class PairedNegSDR(nn.Module):
    r'''
        SDR Loss for sources w/o PIT
        
        input:
            est_targets: (B, nsrc, L)
            targets: (B, nsrc, L)
        
        return:
            (batch)
    '''
    def __init__(self, sdr_type, zero_mean=True, take_log=True, EPS=1e-8, SDR_max=30):
        super(PairedNegSDR, self).__init__()                    
        assert sdr_type in ["snr", "sisdr", "sdsdr", "tsdr", "eps_tsdr"]
        self.sdr_type = sdr_type
        self.zero_mean = zero_mean
        self.take_log = take_log
        self.EPS = EPS
        self.tau = 10 ** (-SDR_max / 10)
        
    def calculate_sdr(self, est_targets, targets):
        if self.sdr_type == "tsdr":
            # [batch, n_src, time]
            e_noise = est_targets - targets
            
            # [batch, n_src]
            pair_wise_sdr = torch.sum(targets**2, dim=2) / (
                torch.sum(e_noise**2, dim=2) + self.tau * torch.sum(targets**2, dim=2)
            )

            return pair_wise_sdr
        
        if self.sdr_type == "eps_tsdr":
            # [batch, n_src, time]    
            e_noise = est_targets - targets
            
            # [batch, n_src]
            pair_wise_sdr = (torch.sum(targets**2, dim=2)  + self.EPS) / (
                torch.sum(e_noise**2, dim=2) + self.tau * (torch.sum(targets**2, dim=2)  + self.EPS)
                )
            
            return pair_wise_sdr
        
        if self.sdr_type in ["sisdr", "sdsdr"]:
            # [batch, n_src, 1]
            pair_wise_dot = torch.sum(est_targets * targets, dim=2, keepdim=True)
            # [batch, n_src, 1]
            s_target_energy = torch.sum(targets**2, dim=2, keepdim=True) + self.EPS
            # [batch, n_src, time]
            pair_wise_proj = pair_wise_dot * targets / s_target_energy
        else:
            # [batch, n_src, time]
            pair_wise_proj = targets
            
        if self.sdr_type in ["sdsdr", "snr"]:
            e_noise = est_targets - targets
        else:
            e_noise = est_targets - pair_wise_proj
            
        # [batch, n_src]
        pair_wise_sdr = torch.sum(pair_wise_proj**2, dim=2) / (
            torch.sum(e_noise**2, dim=2) + self.EPS
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
        
        # [batch, n_src]
        pair_wise_sdr = self.calculate_sdr(est_targets, targets)

        if self.take_log:
            pair_wise_sdr = 10 * torch.log10(pair_wise_sdr + self.EPS)
        
        # [batch]
        return -torch.mean(pair_wise_sdr, dim=1)
    
    
class BinauralLossWrapper(nn.Module):
    def __init__(self, monaural_loss_func):
        super().__init__()
        
        self.monaural_loss_func = monaural_loss_func
        
        
    def forward(self, est_targets_l, est_targets_r, targets_l, targets_r):
        # est_targets_l/targets_l: (batch_size, nsrc, wav_len)
        # est_targets_r/targets_r: (batch_size, nsrc, wav_len)
        
        # (batch_size)
        loss_l = self.monaural_loss_func(est_targets_l, targets_l)
        loss_r = self.monaural_loss_func(est_targets_r, targets_r)
        loss = 0.5 * (loss_l + loss_r)
        
        return loss
        