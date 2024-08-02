import os
import time
import yaml
import models
import logging
import argparse

from datetime import datetime
from functools import partial
from contextlib import suppress
from collections import OrderedDict

from models.softmax_free_vision_transformer import (
    SoftmaxFreeVisionTransformer
)

import torch
import torch.nn as nn
import torchvision.utils

from timm.utils import *
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from timm.loss import (
    LabelSmoothingCrossEntropy, 
    SoftTargetCrossEntropy
)
from timm.data import (
    Dataset, 
    create_loader, 
    resolve_data_config, 
    Mixup, 
    FastCollateMixup
)
from timm.models import (
    load_checkpoint,
    resume_checkpoint
)

# The first arg parser parses out only the --config argument, this argument is used to
# load a yaml file containing key-values that override the defaults for the main parser below
parser = argparse.ArgumentParser(description='SOFT Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')

args = parser.parse_args()

if args.config:
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        parser.set_defaults(**config)
        args = parser.parse_args()
else:
    print("No config file found: please provide path using --config <PATH TO CONFIG>")

args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)

torch.backends.cudnn.benchmark = True
_logger = logging.getLogger('train')

def main():
    setup_default_logging()

    args.prefetcher = not args.no_prefetcher

    args.device = 'cuda:0'
    _logger.info('Training with a single process on %d GPUs.' % args.num_gpu)

    torch.manual_seed(args.seed)

    model = SoftmaxFreeVisionTransformer(
        patch_size=4, 
        embed_dims=[64, 128, 320, 512], 
        num_heads=[2, 4, 10, 16], 
        mlp_ratios=[8, 8, 4, 4], 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        depths=[1, 2, 3, 2], 
        sr_ratios=[8, 4, 2, 1], 
        num_classes=args.num_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path, # stochastic depth
        newton_max_iter=args.newton_maxiter
    )

    args.lr *= (args.batch_size / 1024.0)
    _logger.info("learning rate is %f" % args.lr)

    _logger.info('Model %s created, param count: %d' %
                    (args.model, sum([m.numel() for m in model.parameters()])))

    data_config = resolve_data_config(vars(args), model=model, verbose=True)

    model.cuda()
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    optimizer = create_optimizer(args, model)

    # optionally resume from a checkpoint
    resume_epoch = None
    if args.resume:
        resume_epoch = resume_checkpoint(
            model, args.resume,
            optimizer=None if args.no_resume_opt else optimizer,
            log_info=True
        )

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, 
        # and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume=args.resume
        )
    
    lr_scheduler, num_epochs = create_scheduler(args, optimizer)

    start_epoch = 0
    if args.start_epoch is not None:
        # a specified start_epoch will always override the resume epoch
        start_epoch = args.start_epoch
    elif resume_epoch is not None:
        start_epoch = resume_epoch
    if lr_scheduler is not None and start_epoch > 0:
        lr_scheduler.step(start_epoch)

    _logger.info('Scheduled epochs: {}'.format(num_epochs))

    train_dir = os.path.join(args.data, 'train')
    if not os.path.exists(train_dir):
        _logger.error('Training folder does not exist at: {}'.format(train_dir))
        exit(1)
    dataset_train = Dataset(train_dir)

    collate_fn = None
    mixup_fn = None
    mixup_active = (args.mixup > 0 or \
                    args.cutmix > 0 or \
                    args.cutmix_minmax is not None 
    ) and not args.disable_mix

    if mixup_active:
        mixup_args = dict(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.num_classes)
        if args.prefetcher:
            collate_fn = FastCollateMixup(**mixup_args)
        else:
            mixup_fn = Mixup(**mixup_args)

    loader_train = create_loader(
        dataset_train,
        input_size=data_config['input_size'],
        batch_size=args.batch_size,
        is_training=True,
        use_prefetcher=args.prefetcher,
        no_aug=args.no_aug,
        re_prob=args.reprob,
        re_mode=args.remode,
        re_count=args.recount,
        scale=args.scale,
        ratio=args.ratio,
        hflip=args.hflip,
        vflip=args.vflip,
        color_jitter=args.color_jitter,
        auto_augment=args.aa,
        interpolation=args.train_interpolation,
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        collate_fn=collate_fn,
        pin_memory=args.pin_mem,
        use_multi_epochs_loader=args.use_multi_epochs_loader
    )

    eval_dir = os.path.join(args.data, 'val')
    if not os.path.isdir(eval_dir):
        eval_dir = os.path.join(args.data, 'validation')
        if not os.path.isdir(eval_dir):
            _logger.error('Validation folder does not exist at: {}'.format(eval_dir))
            exit(1)
    dataset_eval = Dataset(eval_dir)

    loader_eval = create_loader(
        dataset_eval,
        input_size=data_config['input_size'],
        batch_size=args.validation_batch_size_multiplier * args.batch_size,
        is_training=False,
        use_prefetcher=args.prefetcher,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        crop_pct=data_config['crop_pct'],
        pin_memory=args.pin_mem,
    )

    _logger.info('Augmentation check: \n Mixup: {} & enabled: {} \n Collate: {}'.format(mixup_active, loader_train.mixup_enabled, collate_fn is not None))
    if mixup_active:
        # smoothing is handled with mixup target transform
        train_loss_fn = SoftTargetCrossEntropy().cuda()
    elif args.smoothing:
        train_loss_fn = LabelSmoothingCrossEntropy(smoothing=args.smoothing).cuda()
    else:
        train_loss_fn = nn.CrossEntropyLoss().cuda()
    validate_loss_fn = nn.CrossEntropyLoss().cuda()

    eval_metric = args.eval_metric
    best_metric = None
    best_epoch = None

    if args.eval:  # evaluate the model
        load_checkpoint(model, args.eval_checkpoint, args.model_ema)
            
        val_metrics = validate(model, loader_eval, validate_loss_fn, args)
        print(f"Top-1 accuracy of the model is: {val_metrics['top1']:.1f}%")
        return

    saver = None
    # output_dir = ''
    output_base = args.output if args.output else './output'
    exp_name = '-'.join([
        datetime.now().strftime("%Y%m%d-%H%M%S"),
        args.model,
        str(data_config['input_size'][-1])
    ])
    output_dir = get_outdir(output_base, 'train', exp_name)
    decreasing = True if eval_metric == 'loss' else False
    saver = CheckpointSaver(
        model=model, optimizer=optimizer, args=args, model_ema=model_ema,
        checkpoint_dir=output_dir, recovery_dir=output_dir, decreasing=decreasing)
    with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
        f.write(args_text)

    try:  # train the model
        for epoch in range(start_epoch, args.epochs):
            train_metrics = train_epoch(
                epoch, model, loader_train, optimizer, train_loss_fn, args,
                lr_scheduler=lr_scheduler, saver=saver, output_dir=output_dir,
                model_ema=model_ema, mixup_fn=mixup_fn)

            eval_metrics = validate(model, loader_eval, validate_loss_fn, args)

            if model_ema is not None and not args.model_ema_force_cpu:
                ema_eval_metrics = validate(
                    model_ema.ema, loader_eval, validate_loss_fn, args, log_suffix=' (EMA)')
                eval_metrics = ema_eval_metrics

            if lr_scheduler is not None:
                # step LR for next epoch
                lr_scheduler.step(epoch + 1, eval_metrics[eval_metric])

            update_summary(
                epoch, train_metrics, eval_metrics, os.path.join(output_dir, 'summary.csv'),
                write_header=best_metric is None)

            if saver is not None:
                # save proper checkpoint with eval metric
                save_metric = eval_metrics[eval_metric]
                best_metric, best_epoch = saver.save_checkpoint(epoch, metric=save_metric)

    except KeyboardInterrupt:
        pass
    if best_metric is not None:
        _logger.info('*** Best metric: {0} (epoch {1})'.format(best_metric, best_epoch))


def train_epoch(
        epoch, model, loader, optimizer, loss_fn, args,
        lr_scheduler=None, saver=None, output_dir='',
        model_ema=None, mixup_fn=None):
    if args.mixup_off_epoch and epoch >= args.mixup_off_epoch:
        if args.prefetcher and loader.mixup_enabled:
            loader.mixup_enabled = False
        elif mixup_fn is not None:
            mixup_fn.mixup_enabled = False

    second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()

    model.train()

    end = time.time()
    last_idx = len(loader) - 1
    num_updates = epoch * len(loader)
    for batch_idx, (input, target) in enumerate(loader):
        last_batch = batch_idx == last_idx
        data_time_m.update(time.time() - end)
        if not args.prefetcher:
            input, target = input.cuda(), target.cuda()
            if mixup_fn is not None:
                input, target = mixup_fn(input, target)
        if args.channels_last:
            input = input.contiguous(memory_format=torch.channels_last)
        output = model(input)
        loss = loss_fn(output, target)

        losses_m.update(loss.item(), input.size(0))

        optimizer.zero_grad()
        loss.backward(create_graph=second_order)
        if args.clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        if args.opt == 'lamb':
            optimizer.step(epoch=epoch)
        else:
            optimizer.step()

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)
        num_updates += 1

        batch_time_m.update(time.time() - end)
        if last_batch or batch_idx % args.log_interval == 0:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)

            _logger.info(
                'Train: {} [{:>4d}/{} ({:>3.0f}%)]  '
                'Loss: {loss.val:>9.6f} ({loss.avg:>6.4f})  '
                'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s  '
                '({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                'LR: {lr:.3e}  '
                'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                    epoch,
                    batch_idx, len(loader),
                    100. * batch_idx / last_idx,
                    loss=losses_m,
                    batch_time=batch_time_m,
                    rate=input.size(0) / batch_time_m.val,
                    rate_avg=input.size(0) / batch_time_m.avg,
                    lr=lr,
                    data_time=data_time_m))

            if args.save_images and output_dir:
                torchvision.utils.save_image(
                    input,
                    os.path.join(output_dir, 'train-batch-%d.jpg' % batch_idx),
                    padding=0,
                    normalize=True)

        if saver is not None and args.recovery_interval and (
                last_batch or (batch_idx + 1) % args.recovery_interval == 0):
            saver.save_recovery(epoch, batch_idx=batch_idx)

        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

        end = time.time()
        # end for

    if hasattr(optimizer, 'sync_lookahead'):
        optimizer.sync_lookahead()

    return OrderedDict([('loss', losses_m.avg)])


def validate(model, loader, loss_fn, args, log_suffix=''):
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()

    model.eval()

    end = time.time()
    last_idx = len(loader) - 1
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            # input = input.type(torch.float64)
            last_batch = batch_idx == last_idx
            if not args.prefetcher:
                input = input.cuda()
                target = target.cuda()
            if args.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)

            output = model(input)
            if isinstance(output, (tuple, list)):
                output = output[0]

            # augmentation reduction
            reduce_factor = args.tta
            if reduce_factor > 1:
                output = output.unfold(0, reduce_factor, reduce_factor).mean(dim=2)
                target = target[0:target.size(0):reduce_factor]

            loss = loss_fn(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            reduced_loss = loss.data

            torch.cuda.synchronize()

            losses_m.update(reduced_loss.item(), input.size(0))
            top1_m.update(acc1.item(), output.size(0))
            top5_m.update(acc5.item(), output.size(0))

            batch_time_m.update(time.time() - end)
            end = time.time()
            if (last_batch or batch_idx % args.log_interval == 0):
                log_name = 'Test' + log_suffix
                _logger.info(
                    '{0}: [{1:>4d}/{2}]  '
                    'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                    'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                    'Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  '
                    'Acc@5: {top5.val:>7.4f} ({top5.avg:>7.4f})'.format(
                        log_name, batch_idx, last_idx, batch_time=batch_time_m,
                        loss=losses_m, top1=top1_m, top5=top5_m))

    metrics = OrderedDict([('loss', losses_m.avg), ('top1', top1_m.avg), ('top5', top5_m.avg)])

    return metrics

if __name__ == '__main__':
    main()
