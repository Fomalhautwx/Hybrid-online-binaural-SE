import os
import toml
from tqdm import tqdm
import numpy as np

import torch
import torch.distributed as dist
from torch.optim import lr_scheduler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset, DataLoader
import logging

from metrics import pesq_scorer, stoi_scorer
from loss_modules.pit_loss import PIT_AlignLossWrapper


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
        self.alpha = config["loss"]["alpha"]
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
            with open(os.path.join(self.exp_path, 'config.toml'), 'w') as f:
                toml.dump(config, f)
            
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
        
        for step, (x, y) in enumerate(tqdm(self.train_dataloader), start=1):
            # x: (bs, 2, L)
            # y: (bs, 2, L)
            x = x.to(self.rank, non_blocking=True)
            y = y.to(self.rank, non_blocking=True)   

            est_r, est_s = self.model(x)
            if isinstance(self.loss_func, PIT_AlignLossWrapper):
                loss_r, loss_s = self.loss_func(est_r, est_s, y, return_est=False)
                
            else:
                loss_r = self.loss_func(est_r, y, return_est=False)
                loss_s = self.loss_func(est_s, y, return_est=False)
            loss = self.alpha * loss_r + (1 - self.alpha) * loss_s
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_grad_norm_value)
            self.optimizer.step()
            
            loss = self._reduce_mean(loss, self.world_size)
            total_loss += loss.item()
            
            if step % self.log_interval == 0 and self.rank == 0:
                log_message = f"epoch [{epoch}/{self.epochs}] [{step}/{len(self.train_dataloader)}]: train_loss={total_loss / step:.4f}"
                self.logging.info(log_message)
                self.writer.add_scalar('Train/loss', total_loss / step, (epoch-1) * dataloader_len + step)


    @torch.no_grad()
    def _validation_epoch(self, epoch):
        total_loss = 0.0
        total_pesq = 0.0
        total_stoi = 0.0
        dataloader_len = len(self.validation_dataloader)
    
        for step, (x, y) in enumerate(tqdm(self.validation_dataloader), start=1):
            # x: (bs, 2, L)
            # y: (bs, 2, L)
            x = x.to(self.rank, non_blocking=True)
            y = y.to(self.rank, non_blocking=True)

            # est: (bs, F, T, 2)
            # y: (bs, F, T, 2)
            est_r, est_s = self.model(x)
            if isinstance(self.loss_func, PIT_AlignLossWrapper):
                loss_r, loss_s, est_reordered = self.loss_func(est_r, est_s, y, return_est=True)
            
            else:
                loss_r = self.loss_func(est_r, y, return_est=False)
                loss_s, est_reordered = self.loss_func(est_s, y, return_est=True)
            loss = self.alpha * loss_r + (1 - self.alpha) * loss_s
            pesq_score = self.pesq(est_reordered, y)
            stoi_score = self.stoi(est_reordered, y)
            
            loss = self._reduce_mean(loss, self.world_size)
            pesq_score = self._reduce_mean(pesq_score, self.world_size)
            stoi_score = self._reduce_mean(stoi_score, self.world_size)
            
            total_loss += loss.item()
            total_pesq += pesq_score.item()
            total_stoi += stoi_score.item()
            
        if self.rank == 0:
            log_message = (f"epoch [{epoch}/{self.epochs}]: valid_loss={total_loss / dataloader_len:.4f}\n"
               f"epoch [{epoch}/{self.epochs}]: pesq={total_pesq / dataloader_len:.3f} | stoi={total_stoi / dataloader_len:.3f}")
            self.logging.info(log_message)
            
            self.writer.add_scalar('Val/loss', total_loss / dataloader_len, epoch)
            self.writer.add_scalar('Val/pesq', total_pesq / dataloader_len, epoch)
            self.writer.add_scalar('Val/stoi', total_stoi / dataloader_len, epoch)

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