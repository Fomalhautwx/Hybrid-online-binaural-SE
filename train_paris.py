import os
import random
import argparse

import numpy as np
import yaml
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

from datasets.datasets_new import Dataloader
from models.Paris_bone import GTCRN_PARIS
# from models.gtcrn_paris_binaural import GTCRN_PARIS_Binaural
from loss_modules.loss import HybridLoss
from trainers.solver_paris import Solver


def init_seeds(seed=0):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(rank, world_size, config, args):
    dist.init_process_group(
        backend="nccl", rank=rank, world_size=world_size,
        init_method="tcp://localhost:29501")

    init_seeds(seed=args.seed + rank)
    torch.cuda.empty_cache()
    torch.cuda.set_device(rank)

    # ---- datasets ----
    train_dataset = Dataloader(**config["train_dataset"])
    validation_dataset = Dataloader(**config["val_dataset"])

    # ---- model ----
    model = GTCRN_PARIS(**config["network"], params=config["network"])

    # 加载预训练权重(用于 BSS finetune), resume 交给 Solver 处理
    pretrained = config.get("pretrained", None)
    if pretrained:
        ckpt_dict = torch.load(pretrained["ckpt"], map_location="cpu")
        model.load_state_dict(ckpt_dict["model"], strict=False)
        if rank == 0:
            print("Resumed pretrained model.")

    model.to(rank)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[rank], output_device=rank,
        find_unused_parameters=False, broadcast_buffers=False)

    # ---- optimizer / loss ----
    optimizer = torch.optim.Adam(params=model.parameters(), **config["optimizer"])
    loss_func = HybridLoss().to(rank)

    writer = SummaryWriter(log_dir=f"{args.save_folder}/logs") if rank == 0 else None

    config["trainer"]["exp_path"] = args.save_folder
    solver = Solver(config=config, model=model, optimizer=optimizer,
                    loss_func=loss_func,
                    train_data=train_dataset, validation_data=validation_dataset,
                    writer=writer, rank=rank, world_size=world_size)

    if args.evaluate_only:
        if rank == 0:
            solver.evaluate()
    else:
        print(f"Rank {rank}: start training")
        solver.train()

    dist.destroy_process_group()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()

    parse = argparse.ArgumentParser()
    parse.add_argument("--cfg_filename", type=str,
                       default="/home/nis/jizhe.lu/Code/DNS3_2Mix_noisy_reverb/config.yaml")
    parse.add_argument("--seed", type=int, default=3407)
    parse.add_argument("--save_folder", default="exps/derev",
                       help="Location to save epoch models")
    parse.add_argument("--evaluate_only", action="store_true")
    args = parse.parse_args()

    with open(args.cfg_filename, "r") as f:
        config = yaml.safe_load(f)
    os.makedirs(args.save_folder, exist_ok=True)

    mp.spawn(main, args=(world_size, config, args), nprocs=world_size, join=True)
