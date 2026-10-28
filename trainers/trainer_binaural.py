import json
import os
from tqdm import tqdm
import numpy as np

import torch
import torch.distributed as dist
from torch.optim import lr_scheduler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset, DataLoader
import logging

from metrics import pesq_scorer, stoi_scorer
from .modules import multich_stft
import cProfile, pstats, io


class Trainer:
    def __init__(self, config, model, optimizer, loss_func,
                 train_dataset, validation_dataset,
                 writer, rank, world_size):
        
        self.rank = rank
        self.world_size = world_size
                
        self.train_sampler = DistributedSampler(train_dataset, shuffle=True)
        self.train_dataloader = DataLoader(train_dataset, sampler=self.train_sampler, **config['train_dataloader'])
        self.validation_sampler = DistributedSampler(validation_dataset, shuffle=False)
        self.validation_dataloader = DataLoader(validation_dataset, sampler=self.validation_sampler, **config['val_dataloader'])
        
        self.model = model
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.pesq = pesq_scorer().to(self.rank)
        self.stoi = stoi_scorer().to(self.rank)
        
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, **config["scheduler"])

        # training config
        self.trainer_config = config['trainer']
        self.epochs = self.trainer_config['epochs']
        self.save_checkpoint_interval = self.trainer_config['save_checkpoint_interval']
        self.log_interval = self.trainer_config['log_interval']
        self.clip_grad_norm_value = self.trainer_config['clip_grad_norm_value']
        
        self.resume = self.trainer_config['resume']
        self.exp_path = self.trainer_config['exp_path']
        self.checkpoint_path = os.path.join(self.exp_path, 'checkpoints')
        self.writer = writer

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
            self.loss_log_path = os.path.join(self.exp_path, 'loss_log.txt')
        self.start_epoch = 1

        if self.resume:
            self._resume_checkpoint()

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
    
    def _train_epoch(self, epoch):
        total_loss = 0.0
        dataloader_len = len(self.train_dataloader)
        
        
        for step, (x, y) in enumerate(tqdm(self.train_dataloader, desc="Processing trainset", dynamic_ncols=True), start=1):
            # x: (bs, 2, L) mixed
            # y: (bs, 2, L) clean
            # pr = cProfile.Profile()
            # pr.enable()
            x = x.to(self.rank, non_blocking=True)
            x = multich_stft(x)
            y = y.to(self.rank, non_blocking=True)
            y = multich_stft(y)

            est = self.model(x)   
            loss_dict = self.loss_func(est, y)   #Hybrif loss of stage 1 and 2

            self.optimizer.zero_grad()
            loss_dict["total"].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_grad_norm_value)
            self.optimizer.step()
            
            loss = self._reduce_mean(loss_dict["total"], self.world_size)
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
                        f"total={loss_dict['total'].item():.4f}, "
                        f"cplx={loss_dict['cplx'].item():.4f}, "
                        f"mag={loss_dict['mag'].item():.4f}, "
                        f"sisnr={loss_dict['sisnr'].item():.4f}\n"
                    )
            # pr.disable()
            # s = io.StringIO()
            # sortby = "cumtime"  # 仅适用于 3.6, 3.7 把这里改成常量了
            # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
            # ps.print_stats(8)
            # print(s.getvalue())
        


    @torch.no_grad()
    def _validation_epoch(self, epoch):
        total_loss = 0.0
        total_pesq = 0.0
        total_stoi = 0.0
        dataloader_len = len(self.validation_dataloader)
    
        for step, (x, y) in enumerate(tqdm(self.validation_dataloader), start=1):
            # x: (bs, 2, L)
            # y_l/y_r: (bs, 2, L)
            x = x.to(self.rank, non_blocking=True)
            x = multich_stft(x)
            y = y.to(self.rank, non_blocking=True)
            y = multich_stft(y)

            # est: (bs, F, T, 2)
            # y: (bs, F, T, 2)
            est = self.model(x) 
            loss_dict = self.loss_func(est, y)

            if step % 20 == 0 and self.rank == 0:
                with open(self.loss_log_path, "a") as f:
                    f.write(
                        f"Validation step {step}, "
                        f"total={loss_dict['total'].item():.4f}, "
                        f"cplx={loss_dict['cplx'].item():.4f}, "
                        f"mag={loss_dict['mag'].item():.4f}, "
                        f"sisnr={loss_dict['sisnr'].item():.4f}\n"
                    )

            
            # est = np.ascontiguousarray(est.detach().cpu())
            # y = np.ascontiguousarray(y.detach().cpu())
            # pesq_score = self.pesq(est, y)
            # stoi_score = self.stoi(est, y)
            
            loss = self._reduce_mean(loss_dict["total"], self.world_size)
            # pesq_score = self._reduce_mean(pesq_score, self.world_size)
            # stoi_score = self._reduce_mean(stoi_score, self.world_size)
            
            total_loss += loss.item()
            # total_pesq += pesq_score.item()
            # total_stoi += stoi_score.item()
            
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

        self.scheduler.step(total_loss / dataloader_len)


    def train(self):
        for epoch in range(self.start_epoch, self.epochs + 1):
            self.train_sampler.set_epoch(epoch)
            
            self.model.train()
            self._train_epoch(epoch)
            dist.barrier()

            self.model.eval()
            self._validation_epoch(epoch)
            dist.barrier()
            
            if self.rank == 0 and epoch % self.save_checkpoint_interval == 0:
                self._save_checkpoint(epoch)


if __name__ == "__main__":
    pass