import os
import random
import numpy as np
import yaml
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

from datasets.datasets_new import Dataloader
from models.gtcrn_binaural import GTCRN_Binaural
from models.gtcrn_binaural_dual import GTCRN_Binaural_Dual
from models.gtcrn_paris_binaural import GTCRN_PARIS_Binaural
from models.gtcrn_paris_mask import GTCRN_PARIS_mask
from loss_modules.loss import HybridLoss


def init_seeds(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) # sets the seed for generating random numbers.
    torch.cuda.manual_seed(seed) # Sets the seed for generating random numbers for the current GPU. It’s safe to call this function if CUDA is not available; in that case, it is silently ignored.
    torch.cuda.manual_seed_all(seed) # Sets the seed for generating random numbers on all GPUs. It’s safe to call this function if CUDA is not available; in that case, it is silently ignored.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(rank, world_size, config, args):
    # init
    # dist.init_process_group(backend="nccl", init_method="tcp://localhost:12127",
    #                         rank=rank, world_size=world_size)
    backend = os.environ.get("DIST_BACKEND", "nccl")
    dist.init_process_group(backend, rank=rank, world_size=world_size, init_method='tcp://localhost:29501')   # 其余参数保持你原来的
    # dist.init_process_group(backend='nccl', init_method='env://') torchrun风格
    
    init_seeds(seed=args.seed+rank)
    torch.cuda.empty_cache()
    torch.cuda.set_device(rank)

    # set datasets
    SSep_Dataset = Dataloader
    
    train_dataset = SSep_Dataset(**config['train_dataset'])
    validation_dataset = SSep_Dataset(**config['val_dataset'])

    Model = GTCRN_Binaural_Dual if config.get("dual_mask", False) else GTCRN_Binaural
    #Model = GTCRN_PARIS_mask
    from trainers.trainer_binaural import Trainer
    from trainers.trainer_paris_binaural import Trainer as Trainer_PARIS

    print(config["network"])
    model = Model(**config["network"])
    
    # load pretrained model for finetuning with BSS
    pretrained = config.get("pretrained", None)
    if pretrained:
        ckpt_path = f"{args.save_folder}/checkpoints/{pretrained['ckpt']}"
        ckpt_dict = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt_dict["model"], strict=False)
    
    # resume training
    trainer_config = config["trainer"]
    if trainer_config["resume"]:
        ckpt_path = f"{args.save_folder}/checkpoints/{trainer_config['ckpt']}"
        ckpt_dict = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt_dict["model"], strict=False)
        print(f"Rank {rank} resume model from {ckpt_path}")
    model.to(rank)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[rank], output_device=rank,
        find_unused_parameters=False, broadcast_buffers=False)

    # opt
    optimizer = torch.optim.Adam(params=model.parameters(), **config['optimizer'])

    # loss
    loss = HybridLoss()
    loss.to(rank)
    
    writer = SummaryWriter(log_dir=f"{args.save_folder}/logs") if rank==0 else None
    print("Start training")
    
    config["trainer"]["exp_path"] = args.save_folder
    trainer = Trainer(config=config, model=model, optimizer=optimizer, loss_func=loss,
                       train_dataset=train_dataset, validation_dataset=validation_dataset, 
                       writer=writer, rank=rank, world_size=world_size)
    #trainer = Trainer_PARIS(config=config, model=model, optimizer=optimizer, loss_func=loss,
                    #   train_dataset=train_dataset, validation_dataset=validation_dataset, 
                    #   writer=writer, rank=rank, world_size=world_size)
    debug_path = "/home/nis/jizhe.lu/Code/DNS3_2Mix_noisy_reverb/output/paris_debug"
    os.makedirs(debug_path, exist_ok=True)
    trainer.train()
    dist.destroy_process_group()


if __name__ == '__main__':  
    import argparse
     
    world_size = torch.cuda.device_count()
    
    parse = argparse.ArgumentParser()
    parse.add_argument('--cfg_filename', type=str, default="/home/nis/jizhe.lu/Code/DNS3_2Mix_noisy_reverb/config.yaml")
    
    parse.add_argument('--seed', type=int, default=3407)
    
    parse.add_argument('--paris', action="store_true")

    parse.add_argument('--save_folder', default="exps/derev", help='Location to save epoch models')

    
    args = parse.parse_args()

    config_file = args.cfg_filename
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    os.makedirs(args.save_folder, exist_ok=True)
    mp.spawn(
        main, args=(world_size, config, args,), nprocs=world_size, join=True)
    #main(world_size, config, args)
