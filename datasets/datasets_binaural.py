import soundfile as sf
import torch
import numpy as np
import random
import json
from torch.utils.data import Dataset, DataLoader

from typing import List, Union


EPS = np.finfo(float).eps

class SSep_Dataset_binaural(Dataset):
    def __init__(self, 
                 md_file: Union[str, List[str]],
                 drb: str = "full",
                 wav_len: int = 8,
                 fs: int = 16000,
                 random_start_point: int = True,
                 dummy: bool = False):
        
        if isinstance(md_file, List):
            self.md_data = []
            for md_file_ in md_file:
                self.md_data += json.load(open(md_file_))
                
        else:
            self.md_data = json.load(open(md_file))
            
        self.wav_len = wav_len * fs
        self.fs = fs
        self.random_start_point = random_start_point
        self.drb = drb
        
        if dummy:
            self.md_data = self.md_data[0: len(self.md_data) // 10]
        
    def _get_start_stop(self, filepath):
        if self.wav_len == 0:
            start = 0
            stop = None
        
        else:
            wav_len = sf.info(filepath).frames
            if self.random_start_point:
                start = random.randint(0, wav_len - self.wav_len)
                stop = start + self.wav_len
            else:
                start = 0
                stop = self.wav_len
            
        return start, stop

    def __getitem__(self, idx):
        md_item = self.md_data[idx]
            
        mix_path = md_item["mix_noisy_path"].replace(".wav", "_full.wav")
        tgt_spk1_path = md_item["tgt_spk1_path"].replace(".wav", f"_{self.drb}.wav")
        tgt_spk2_path = md_item["tgt_spk2_path"].replace(".wav", f"_{self.drb}.wav")
        
        start, stop = self._get_start_stop(mix_path)
        mix_data, _ = sf.read(mix_path, dtype="float32", start=start, stop=stop)
        tgt_spk1_data, _ = sf.read(tgt_spk1_path, dtype="float32", start=start, stop=stop)
        tgt_spk2_data, _ = sf.read(tgt_spk2_path, dtype="float32", start=start, stop=stop)
        
        # (2, wav_len)
        mix = torch.from_numpy(mix_data.T)
        tgt_spk1 = torch.from_numpy(tgt_spk1_data.T)
        tgt_spk2 = torch.from_numpy(tgt_spk2_data.T)
        
        # (2, wav_len)
        tgt_l = torch.stack([tgt_spk1[0], tgt_spk2[0]], dim=0)
        tgt_r = torch.stack([tgt_spk1[1], tgt_spk2[1]], dim=0)

        return mix, tgt_l, tgt_r

    def __len__(self):
        return len(self.md_data)



if __name__=='__main__':
    import soundfile as sf
    import os
    from tqdm import tqdm
    
    train_dataset = SSep_Dataset_binaural(
        md_file="/apdcephfs/share_1316500/tianchisun/data/DNS3_2Mix/DNS3_2Mix_v4/metadata/sets/train.json",
        wav_len=8, drb="200ms_truncate", dummy=False)
    print(len(train_dataset))
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    for i, (mix, tgt_l, tgt_r) in tqdm(enumerate(train_loader)):
        print(i, mix.shape, mix.dtype, tgt_l.shape, tgt_l.dtype, tgt_r.shape, tgt_r.dtype)
        if i==10:
            break
        
    val_dataset = SSep_Dataset_binaural(
        md_file="/apdcephfs/share_1316500/tianchisun/data/DNS3_2Mix/DNS3_2Mix_v4/metadata/sets/dev.json",
        wav_len=0, drb="200ms_truncate", dummy=True)
    print(len(val_dataset))
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    for i, (mix, tgt_l, tgt_r) in tqdm(enumerate(val_loader)):
        print(i, mix.shape, mix.dtype, tgt_l.shape, tgt_l.dtype, tgt_r.shape, tgt_r.dtype)
        if i==10:
            break