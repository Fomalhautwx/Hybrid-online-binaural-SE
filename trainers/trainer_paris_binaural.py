'''For PROPOSED PARIS model training.'''
import os
import json
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.optim import lr_scheduler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset, DataLoader
import logging

from metrics import pesq_scorer, stoi_scorer
from loss_modules.pit_loss import PIT_AlignLossWrapper
from .modules import STFT, ISTFT, multich_stft
import soundfile as sf
import cProfile, pstats, io
from utils.train_guard import TrainGuard

def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(
            m.weight,
            a=0.25,
            mode="fan_in",
            nonlinearity="leaky_relu"
        )
        if m.bias is not None:
            nn.init.zeros_(m.bias)

    elif isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(
            m.weight,
            a=0.25,
            mode="fan_out",      # 👈 不同点在这
            nonlinearity="leaky_relu"
        )
        if m.bias is not None:
            nn.init.zeros_(m.bias)

    elif isinstance(m, nn.PReLU):
        nn.init.constant_(m.weight, 0.25)

    elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


class Trainer:
    def __init__(self, config, model, optimizer, scheduler, loss_func,
                 train_dataset, validation_dataset, finetune, 
                 writer, rank, world_size, win_len=512, hop_len=256):
        
        self.rank = rank
        self.world_size = world_size
        self.stft = STFT(win_len, hop_len)
        self.istft = ISTFT(win_len, hop_len)
                
        self.train_sampler = DistributedSampler(train_dataset, shuffle=True)
        self.train_dataloader = DataLoader(train_dataset, sampler=self.train_sampler, **config['train_dataloader'])
        self.validation_sampler = DistributedSampler(validation_dataset, shuffle=False)
        self.validation_dataloader = DataLoader(validation_dataset, sampler=self.validation_sampler, **config['val_dataloader'])
        
        self.model = model
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.alpha = config["loss"]["alpha"]
        self.pesq = pesq_scorer().to(self.rank)
        self.stoi = stoi_scorer().to(self.rank)
        self.scheduler = scheduler
        #self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, **config["scheduler"])
        

        # training config
        self.trainer_config = config['trainer']
        self.epochs = self.trainer_config['epochs']
        self.cl_epochs = self.trainer_config['cl_epochs']
        self.save_checkpoint_interval = self.trainer_config['save_checkpoint_interval']
        self.log_interval = self.trainer_config['log_interval']
        self.clip_grad_norm_value = self.trainer_config['clip_grad_norm_value']
        
        self.resume = self.trainer_config['resume']
        self.exp_path = self.trainer_config['exp_path']
        self.checkpoint_path = os.path.join(self.exp_path, 'checkpoints')
        self.writer = writer
        self.finetune = finetune
        self.loss_log_path = os.path.join(self.exp_path, 'loss.txt')
        self.guard = TrainGuard(self.model, self.optimizer, dump_dir='train_dumps', max_grad_norm=self.clip_grad_norm_value)

        # save the config
        if self.rank == 0:
            os.makedirs(self.checkpoint_path, exist_ok=True)
            with open(os.path.join(self.exp_path, 'config.json'), 'w') as f:
                json.dump(config, f, ensure_ascii=False, indent=4)
            
            logging.basicConfig(
                filename=os.path.join(self.exp_path, 'log.txt'),
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s',
                filemode="a"
                )
            self.logging = logging.getLogger()
            
        self.start_epoch = 1

        if self.resume:
            self._resume_checkpoint()
        # 子网络初始化已移至 train.py：在加载任何权重之前、只作用于 encoder/dpgrnn1/dpgrnn2/decoder。
        # 此处不能再 apply(init_weights)——会把 teacher/DNN-WPE 预训练权重和已加载的微调权重洗掉。
            
        if self.finetune:
            n_bn = self.recalibrate_bn(self.model, train_dataset,
                                  config['train_dataloader']['batch_size'],
                                  self.rank, config.get("finetune", {}).get("bn_recalib_batches", 0))
            dist.barrier()
            
            if self.rank == 0 and n_bn:
                print(f"[finetune] BN 重校准完成，共 {n_bn} 个 BN 层")

    def reset_buffer(self):
        self.model.module.validate_reset_buffer()

    def _save_checkpoint(self, epoch):
        state_dict = {
            'epoch': epoch,
            'model': self.model.module.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()}

        torch.save(state_dict, os.path.join(self.checkpoint_path, f'model_epoch{epoch}.tar'))

    def _resume_checkpoint(self):
        # latest_checkpoints = sorted(glob(os.path.join(self.checkpoint_path, 'model_*.tar')))[-1]
        latest_checkpoints  = f"{self.checkpoint_path}/{self.trainer_config['ckpt']}"
        checkpoint = torch.load(latest_checkpoints, map_location=torch.device(self.rank))

        self.start_epoch = checkpoint['epoch'] + 1
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        # self.model.module.load_state_dict(checkpoint['model'])
        
        if self.rank == 0:
            log_message = f"resume checkpoint from epoch {self.start_epoch}"
            self.logging.info(log_message)
        
    @torch.no_grad()
    def _reduce_mean(self, tensor, nprocs):
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        rt /= nprocs
        return rt
    
    @torch.no_grad()
    def recalibrate_bn(self, model, dataset, batch_size, device, num_batches=50):
        """用新前端的数据重新估计 BN running stats。
        必须在 DDP 包装之后调用，SyncBatchNorm 会跨卡同步统计量。"""
        bns = [m for m in model.modules()
            if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d,
                                torch.nn.SyncBatchNorm))]
        if not bns or num_batches <= 0:
            return 0

        saved_momentum = [m.momentum for m in bns]
        for m in bns:
            m.reset_running_stats()
            m.momentum = None            # 累积平均，比指数平均更准

        loader = DataLoader(dataset, batch_size=batch_size, num_workers=4,
                            shuffle=True, drop_last=True)
        was_training = model.training
        model.train()
        for i, (x, y) in enumerate(tqdm(loader, total=num_batches)):
            x = x.to(device, non_blocking=True)
            x = multich_stft(x)
            y = y.to(device, non_blocking=True)
            y = multich_stft(y)
            model(x, y)                     # ★ 改成与 Trainer 里完全一致的调用方式
            if i + 1 >= num_batches:
                break
        if not was_training:
            model.eval()
        for m, mom in zip(bns, saved_momentum):
            m.momentum = mom
        return len(bns)
    
    def _train_epoch(self, epoch):
        total_loss = 0.0
        dataloader_len = len(self.train_dataloader)
        
        for step, (x_in, y) in enumerate(tqdm(self.train_dataloader, desc="Processing trainset", dynamic_ncols=True), start=1):
            # x: (bs, 2, L) mixed
            # y: (bs, 2, L) clean
            # pr = cProfile.Profile()
            # pr.enable()
            
            cl_factor = None
            x_in = x_in.to(self.rank, non_blocking=True)
            # sf.write(f"/home/nis/jizhe.lu/Code/DNS3_2Mix_noisy_reverb/output/paris_debug/noisy_{step}.wav", x.squeeze().T.cpu().numpy(), 16000)
            x = multich_stft(x_in)
            y = y.to(self.rank, non_blocking=True)
            y = multich_stft(y)
            
            # Curriculum learning
            if epoch <= self.cl_epochs:
                cl_factor = dataloader_len * (epoch - 1) + step
                cl_factor = cl_factor / (self.cl_epochs * dataloader_len)
                self.model.module.set_cl_factor(cl_factor)
            
            est, dummy_out = self.model(x, y)   # 已加上clean_prior\
            loss1 = self.loss_func(dummy_out, y)
            loss2 = self.loss_func(est, y)
            loss = self.alpha * loss1["total"] + (1 - self.alpha) * loss2["total"]    #Hybrif loss of stage 1 and 2
            
            # debug
            # est = torch.istft(est.squeeze(), n_fft=512, 
            #          hop_length=256, 
            #          win_length=512, 
            #          window=torch.hann_window(512).pow(0.5).cuda())
            # sf.write(f"/home/nis/jizhe.lu/Code/DNS3_2Mix_noisy_reverb/output/paris_debug/output_{step}.wav", est.detach().T.cpu().numpy(), 16000)
            
            if not self.guard.step(loss, batch=None, step=step):
                sf.write(f"/home/nis/jizhe.lu/Code/DNS3_2Mix_noisy_reverb/train_dump/bad_batch_{step}.wav", x_in.squeeze().T.cpu().numpy(), 16000)
                torch.save({"X":x, "prior":y}, f"/home/nis/jizhe.lu/Code/DNS3_2Mix_noisy_reverb/train_dump/debug{step}.pt")
                self.guard.summary()
                

            # self.optimizer.zero_grad()
            # loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_grad_norm_value)
            # self.optimizer.step()
            self.scheduler.step()
            
            # pr.disable()
            # s = io.StringIO()
            # sortby = "cumtime"  # 仅适用于 3.6, 3.7 把这里改成常量了
            # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
            # ps.print_stats(35)
            # print(s.getvalue())
            
            loss = self._reduce_mean(loss, self.world_size)
            total_loss += loss.item()
            lr = self.scheduler._last_lr[0]
            
            if step % self.log_interval == 0 and self.rank == 0:
                log_message = f"epoch [{epoch}/{self.epochs}] [{step}/{len(self.train_dataloader)}]: train_loss={total_loss / step:.4f}"
                self.logging.info(log_message)
                self.writer.add_scalar('Train/loss', total_loss / step, (epoch-1) * dataloader_len + step)
                self.writer.add_scalar('Train/lr', lr, (epoch-1) * dataloader_len + step)
                with open(self.loss_log_path, "a") as f:
                    f.write(
                        f"{step}, "
                        f"total={loss1['total'].item():.4f}+{loss2['total'].item():.4f}, "
                        f"cplx={loss2['cplx'].item():.4f}, "
                        f"mag={loss2['mag'].item():.4f}, "
                        f"sisnr={loss2['sisnr'].item():.4f}, "
                        f"{cl_factor}\n"
                    )


    @torch.no_grad()
    def _validation_epoch(self, epoch):
        total_loss = torch.tensor([0.0]).cuda()
        dataloader_len = len(self.validation_dataloader)
        print(">>> Validation dataset size:", dataloader_len)
    
        for step, (x, y) in enumerate(tqdm(self.validation_dataloader), start=1):
            # x: (bs, 2, L)
            # y_l/y_r: (bs, 2, L)
            x = x.to(self.rank, non_blocking=True)
            x = multich_stft(x)
            y = y.to(self.rank, non_blocking=True)
            y = multich_stft(y)
            
            
            est, dummy_out = self.model(x, y)
            loss1 = self.loss_func(dummy_out, y)
            loss2 = self.loss_func(est, y)
            loss = self.alpha * loss1["total"] + (1 - self.alpha) * loss2["total"]

            if step % 100 == 0 and self.rank == 0:
                with open(self.loss_log_path, "a") as f:
                    f.write(
                        f"Validation epoch {epoch} step {step}, "
                        f"total={loss2['total'].item():.4f}, "
                        f"cplx={loss2['cplx'].item():.4f}, "
                        f"mag={loss2['mag'].item():.4f}, "
                        f"sisnr={loss2['sisnr'].item():.4f}\n"
                    )
            # est = np.ascontiguousarray(est.detach().cpu())
            # y = np.ascontiguousarray(y.detach().cpu())
            # pesq_score = self.pesq(est, y)
            # stoi_score = self.stoi(est, y)
            
            loss = self._reduce_mean(loss, self.world_size)
            # pesq_score = self._reduce_mean(pesq_score, self.world_size)
            # stoi_score = self._reduce_mean(stoi_score, self.world_size)
            total_loss += loss
            
            # total_pesq += pesq_score.item()
            # total_stoi += stoi_score.item()
        total_loss = total_loss.item()
        # if self.rank == 0:
        #     log_message = (f"epoch [{epoch}/{self.epochs}]: valid_loss={total_loss / dataloader_len:.4f}\n"
        #        f"epoch [{epoch}/{self.epochs}]: pesq={total_pesq / dataloader_len:.3f} | stoi={total_stoi / dataloader_len:.3f}")
        #     self.logging.info(log_message)
        if self.rank == 0:
            log_message = (f"epoch [{epoch}/{self.epochs}]: valid_loss={total_loss / dataloader_len:.4f}\n")
            self.logging.info(log_message)
            
            self.writer.add_scalar('Val/loss', total_loss / dataloader_len, epoch)
            # self.writer.add_scalar('Val/pesq', total_pesq / dataloader_len, epoch)
            # self.writer.add_scalar('Val/stoi', total_stoi / dataloader_len, epoch)

        #self.scheduler.step(total_loss / dataloader_len)


    def train(self):
        for epoch in range(self.start_epoch, self.epochs + 1):
            self.train_sampler.set_epoch(epoch)
            
            print("TRAINING EPOCH {}".format(epoch))
            self.model.train()
            self._train_epoch(epoch)
            dist.barrier()

            print("VALIDATING")

            self.model.eval()
            self._validation_epoch(epoch)
            
            dist.barrier()
            
            
            if self.rank == 0 and epoch % self.save_checkpoint_interval == 0:
                self._save_checkpoint(epoch)


if __name__ == "__main__":
    pass