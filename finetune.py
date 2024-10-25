############################################################
# Code for FiT3D 
# by Yuanwen Yue
# Stage 2: 3D-aware fine-tuning
############################################################

import argparse
import datetime
import json
import random
import os
import time
from pathlib import Path

import numpy as np
import wandb
import torch
from torch.utils.data import DataLoader
import utils.misc as utils
from datasets import build_dataset
from engine import evaluate_one_epoch, train_one_epoch

from utils.loss_utils import l1_loss
from utils.model_utils import build_2d_model

def get_args_parser():
    parser = argparse.ArgumentParser('FiT3D', add_help=False)
    parser.add_argument('--lr', default=1e-5, type=float)

    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--lr_drop', default=[1], type=list)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')


    parser.add_argument("--model_name", default='dinov2_small', type=str, help='2D feature extractor. Select from \
                                        dinov2_small, dinov2_base, dinov2_reg_small, clip_base, mae_base, deit3_base')
    parser.add_argument('--output_dir', default='output_finemodel',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--job_name', default='finetuning_dinov2_small', type=str)
    
    parser.add_argument('--dataset_root', default='db/scannetpp/scenes', type=str)
    parser.add_argument('--train_list', default='db/scannetpp/metadata/train_samples_all.txt', type=str)
    parser.add_argument('--val_list', default='db/scannetpp/metadata/val_samples_all.txt', type=str)
    parser.add_argument('--train_gaussian_list', default='db/scannetpp/metadata/pretrained_feat_gaussians_train.pth', type=str)
    parser.add_argument('--val_gaussian_list', default='db/scannetpp/metadata/pretrained_feat_gaussians_val.pth', type=str)
    parser.add_argument('--train_view_list', default='db/scannetpp/metadata/train_view_info.npy', type=str)
    parser.add_argument('--val_view_list', default='db/scannetpp/metadata/val_view_info.npy', type=str)
   
    return parser



def main(args):

    print(args)

    # setup wandb for logging
    utils.setup_wandb()
    wandb.init(project="FiT3D")
    wandb.run.name = args.run_name

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # build model
    model = build_2d_model(model_name=args.model_name)
    model.to(device)

    # loss
    criterion = l1_loss

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    # build dataset and dataloader
    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)


    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    def trivial_batch_collator(batch):
        """
        A batch collator that does nothing.
        """
        return batch

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=trivial_batch_collator, num_workers=args.num_workers,
                                   pin_memory=True)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=trivial_batch_collator, num_workers=args.num_workers,
                                 pin_memory=True)


    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.lr_drop)

    output_dir = Path(args.output_dir)


    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model'], strict=False)
        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        if len(missing_keys) > 0:
            print('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            print('Unexpected Keys: {}'.format(unexpected_keys))
        if 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            import copy
            p_groups = copy.deepcopy(optimizer.param_groups)
            optimizer.load_state_dict(checkpoint['optimizer'])
            for pg, pg_old in zip(optimizer.param_groups, p_groups):
                pg['lr'] = pg_old['lr']
                pg['initial_lr'] = pg_old['initial_lr']
            print(optimizer.param_groups)
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            lr_scheduler.step(lr_scheduler.last_epoch)
            args.start_epoch = checkpoint['epoch'] + 1
        # check the resumed model
        val_stats = evaluate_one_epoch(
            model, criterion, data_loader_val, device
        )

    print("Start training")
    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch, args.clip_max_norm
            )
        lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            if (epoch + 1) in args.lr_drop or (epoch + 1) % 1 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:

                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        val_stats = evaluate_one_epoch(
            model, criterion, data_loader_val, device
        )

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in val_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}
        
        wandb.log({"lr_rate": train_stats['lr']})

        train_log_dict = {
                "train/epoch": epoch,
                "train/loss_epoch": train_stats['loss'],
                }
  
        val_log_dict = {
                "val/loss": val_stats['loss'],
                }
                
        wandb.log(train_log_dict)
        wandb.log(val_log_dict)

        if args.output_dir:
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('3D-aware fine-tuning script', parents=[get_args_parser()])
    args = parser.parse_args()
    now = datetime.datetime.now()
    run_id = now.strftime("%Y-%m-%d-%H-%M-%S")
    args.run_name = run_id+'_'+args.job_name 
    args.output_dir = os.path.join(args.output_dir, args.run_name)

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
