import soundfile as sf
import torch
import numpy as np
import random
import json
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from typing import List, Union


EPS = np.finfo(float).eps

def _read_scp(scp_path: str, set_type:str, shuffle: bool = False):
    """
    :param scp_path:
    :param set_type:
    :param mic_order:
    :param shuffle:
    :return:
    """
    out_list = list()
    if isinstance(scp_path, List):
            in_list = []
            for md_file_ in scp_path:
                in_list += json.load(open(md_file_, "r"))   # extend
    else:
        in_list = json.load(open(scp_path, "r"))

    if shuffle:
        np.random.shuffle(in_list)

    for line in tqdm(in_list, desc="Loading {} set metadata".format(set_type)):
        """
        每行的数据格式定义如下:
        {
            "rt": 0.3910885061964363,
            "room_dim": [],
            "mic_pos": [],
            "n_interf": 2,
            "target_pos": [],
            "target_angle": 90,
            "interf0_pos": [],
            "interf1_pos": [],
            "snr": 14.616706775524602,
            "clean": "/data/hdd0/jizhe.lu/clean_testset_wav/p232_341.wav",
            "noisy": "/data/hdd0/jizhe.lu/Simulated/sample_0_p232_341_test.wav"
            }
        """
        line_dict = line
        cur_line = dict()
        cur_line["noisy"] = line_dict["noisy"]
        cur_line["target"] = line_dict["RTS"]
        out_list.append(cur_line)
        
    return out_list

class Dataloader(Dataset):
    def __init__(self, 
                 mode: str,
                 md_file: Union[str, List[str]],
                 drb: str = "full",
                 wav_len: int = 8,
                 fs: int = 16000,
                 random_start_point: int = True,
                 dummy: bool = False):
        self.md_data = _read_scp(md_file, set_type="train", shuffle=True)   # return list of noisy and clean path
        
        self.wav_len = wav_len * fs
        self.fs = fs
        self.random_start_point = random_start_point
        self.drb = drb
        self.mode = mode
        
        if dummy:
            self.md_data = self.md_data[0: len(self.md_data) // 10]
        
    def _get_start_stop(self, filepath):
        if self.wav_len == 0:
            start = 0
            stop = None
        
        else:
            len = sf.info(filepath).frames
            if self.random_start_point and (len > self.wav_len):
                start = random.randint(0, len - self.wav_len)
                stop = start + self.wav_len
            else:
                start = 0
                stop = self.wav_len
            
        return start, stop

    def __getitem__(self, idx):
        md_item = self.md_data[idx]
        mix_path = md_item["noisy"]
        tgt_path = md_item["target"]  # 这里的tgt_path是clean路径
        mix_data, _ = sf.read(mix_path, dtype="float32")    # (wav_len, 2)
        tgt_data, _ = sf.read(tgt_path, dtype="float32")

        if self.wav_len == 0:
            return torch.from_numpy(mix_data.T), torch.from_numpy(tgt_data.T)
        
        repeat = 1
        if len(mix_data) < self.wav_len:
            repeat = self.wav_len // len(mix_data) + 1
            mix_data = np.concatenate([mix_data] * repeat, axis=0)
            tgt_data = np.concatenate([tgt_data] * repeat, axis=0)
        
        total_len = len(mix_data)
        if self.mode == "train":
            start = random.randint(0, total_len - self.wav_len) # 听fable地不用np.random了，去掉+1以免出问题
        elif self.mode == "val":
            start = (total_len - self.wav_len) // 2
        else:
            start = 0
        stop = start + self.wav_len

        mix_data = mix_data[start:stop]
        tgt_data = tgt_data[start:stop]
        # (2, wav_len)
        mix = torch.from_numpy(mix_data.T)
        tgt_spk = torch.from_numpy(tgt_data.T)
        return mix, tgt_spk

    def __len__(self):
        return len(self.md_data)
        #return 17



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