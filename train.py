import os
import random
import numpy as np
import yaml
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
import math
from torch.optim.lr_scheduler import LambdaLR

from datasets.datasets_new import Dataloader

from models.gtcrn_paris_binaural import GTCRN_PARIS_Binaural
from models.Paris_enhancement import GTCRN_PARIS
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
    
def build_warmup_cosine(optimizer, warmup_steps, total_steps, min_lr_ratio=0.01):
    """按 step 调用。warmup 线性升，之后 cosine 退火到 base_lr*min_lr_ratio。"""
    def fn(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        cos = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cos
    return LambdaLR(optimizer, fn)



def main(rank, world_size, config, args):
    # init
    torch.cuda.set_device(rank)
    backend = os.environ.get("DIST_BACKEND", "nccl")
    dist.init_process_group(backend, rank=rank, world_size=world_size, init_method='tcp://localhost:29501')   # 其余参数保持你原来的
    # dist.init_process_group(backend='nccl', init_method='env://') torchrun风格
    
    init_seeds(seed=args.seed+rank)
    torch.cuda.empty_cache()
    

    # set datasets
    SSep_Dataset = Dataloader
    
    train_dataset = SSep_Dataset(**config['train_dataset'])
    validation_dataset = SSep_Dataset(**config['val_dataset'])

    Model = GTCRN_PARIS
    #Model = GTCRN_PARIS_Binaural
    from trainers.trainer_paris_binaural import Trainer

    print(config["network"])
    model = Model(**config["network"], params=config["network"])

    # 从零训练：只初始化子网络本体（可训练部分），必须在加载任何权重之前。
    # 不能对整模 apply(init_weights)（会洗掉 teacher/DNN-WPE 预训练权重），
    # 也不能放在 Trainer 里（那时 pretrained/resume 已加载完，会把微调权重洗掉）。
    if not config["trainer"]["resume"]:
        from trainers.trainer_paris_binaural import init_weights
        for name in ["encoder", "dpgrnn1", "dpgrnn2", "decoder"]:
            getattr(model, name).apply(init_weights)

    # load pretrained model for finetuning with BSS
    pretrained = config.get("pretrained", None)
    if pretrained:
        ckpt_path = pretrained['ckpt']
        ckpt_dict = torch.load(ckpt_path, map_location="cpu")
        # 过滤 derev 前缀：历史 ckpt 里存的是被 init_weights 洗过的 DNN-WPE 权重，
        # 直接加载会把 __init__ 现读的新鲜 WPENet 覆盖掉。
        # derev 所有参数 requires_grad=False、从未被训练，过滤永远无损。
        ckpt_sd = {k: v for k, v in ckpt_dict["model"].items() if not k.startswith('derev.')}
        missing, unexpected = model.load_state_dict(ckpt_sd, strict=False)
        print(f"missing={len(missing)}, unexpected={len(unexpected)}")
        print("unexpected 样例:", missing[:5])
        print("Resumed pretrained model.")

    # resume training
    trainer_config = config["trainer"]
    if trainer_config["resume"]:
        ckpt_path = f"{args.save_folder}/checkpoints/{trainer_config['ckpt']}"
        ckpt_dict = torch.load(ckpt_path, map_location="cpu")
        ckpt_sd = {k: v for k, v in ckpt_dict["model"].items() if not k.startswith('derev.')}
        model.load_state_dict(ckpt_sd, strict=False)
        print(f"Rank {rank} resume model from {ckpt_path}")
    model.to(rank)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[rank], output_device=rank,
        find_unused_parameters=False, broadcast_buffers=False)

    # opt
    optimizer = torch.optim.Adam(params=model.parameters(), **config['optimizer'])
    # 每 rank 的每 epoch 步数 = 该 rank 的 dataloader 长度
    steps_per_epoch = math.ceil(
        len(train_dataset) / (config['train_dataloader']['batch_size'] * world_size))
    total_steps = config['trainer']['epochs'] * steps_per_epoch

    sch_cfg = config.get('scheduler', {})
    scheduler = build_warmup_cosine(
        optimizer,
        warmup_steps=sch_cfg.get('warmup_steps', 500),
        total_steps=total_steps,
        min_lr_ratio=sch_cfg.get('min_lr_ratio', 0.01))

    if rank == 0:
        print(f"[sched] warmup+cosine: {steps_per_epoch} steps/epoch, "
            f"total {total_steps}, peak lr {config['optimizer']['lr']}")
        

    # loss
    loss = HybridLoss()
    loss.to(rank)
    
    writer = SummaryWriter(log_dir=f"{args.save_folder}/logs") if rank==0 else None
    print("Start training")
    
    config["trainer"]["exp_path"] = args.save_folder
    trainer = Trainer(config=config, model=model, optimizer=optimizer, scheduler=scheduler, loss_func=loss,
                      train_dataset=train_dataset, validation_dataset=validation_dataset, finetune=False,#pretrained is not None,
                      writer=writer, rank=rank, world_size=world_size)
    
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
