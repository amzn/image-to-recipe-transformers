# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# -*- coding: utf-8 -*-
"""
Training script
"""
import os
import random
from collections import defaultdict
import numpy as np
import pickle
import torch
import torch.backends.cudnn as cudnn
from torch import nn
from torch.cuda.amp import GradScaler, autocast
import gc
from torch.utils.tensorboard import SummaryWriter
from utils.utils import *
from utils.loss import TripletLoss
from dataset import get_loader
from config import get_args
from models import get_model
from eval import computeAverageMetrics

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAP_LOC = None if torch.cuda.is_available() else 'cpu'


def trainIter(args, split, loader, model, optimizer,
              loss_function, scaler, metrics_epoch):
    with autocast():
        img, title, ingrs, instrs, _ = loader.next()

        img = img.to(device) if img is not None else None
        title = title.to(device)
        ingrs = ingrs.to(device)
        instrs = instrs.to(device)

        if split == 'val':
            with torch.no_grad():
                img_feat, recipe_feat, proj_feats, = model(img, title, ingrs, instrs)
        else:
            out = model(img, title, ingrs, instrs,
                        freeze_backbone=args.freeze_backbone)

            img_feat, recipe_feat, proj_feats = out

        loss_recipe, loss_paired = 0, 0

        if args.recipe_loss_weight > 0:
            # compute triplet loss on pairs of raw and projected
            # feature vectors for all recipe component pairs

            # count number of component pairs for averaging
            c = 0
            names = ['title', 'ingredients', 'instructions']
            # for every recipe component
            for raw_name in names:
                # get the original feature (not projected) as the query (q)
                q = proj_feats['raw'][raw_name]
                # for every other recipe component (proj_name)
                for proj_name in names:
                    if proj_name != raw_name:
                        # get the projection from its raw feature
                        # to the query recipe component as value (v)
                        # (e.g. query=title, value=proj_ingredient2title(ingredient))
                        v = proj_feats[proj_name][raw_name]
                        loss_recipe += loss_function(q, v)
                        c += 1
            loss_recipe /= c
            metrics_epoch['loss_recipe'].append(loss_recipe.item())

        if img is not None:
            loss_paired = loss_function(img_feat, recipe_feat)
            metrics_epoch['loss_paired'].append(loss_paired.item())

        loss = loss_paired + args.recipe_loss_weight*loss_recipe

        metrics_epoch['loss'].append(loss.item())

    if split == 'train':
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    if img_feat is not None:
        img_feat = img_feat.cpu().detach().numpy()
    recipe_feat = recipe_feat.cpu().detach().numpy()

    return img_feat, recipe_feat, metrics_epoch


def train(args):

    checkpoints_dir = os.path.join(args.save_dir, args.model_name)
    make_dir(checkpoints_dir)

    if args.tensorboard:
        logger = SummaryWriter(checkpoints_dir)

    loaders = {}

    if args.resume_from != '':
        print("Resuming from checkpoint: ", args.resume_from)
        # update arguments when loading model
        vars_to_replace = ['batch_size', 'tensorboard',
                           'model_name', 'lr',
                           'scale_lr', 'freeze_backbone',
                           'load_optimizer']
        store_dict = {}
        for var in vars_to_replace:
            store_dict[var] = getattr(args, var)

        resume_path = os.path.join(args.save_dir, args.resume_from)
        args, model_dict, optim_dict = load_checkpoint(resume_path,
                                                       'curr', MAP_LOC,
                                                       store_dict)

        # load current state of training
        curr_epoch = args.curr_epoch
        best_loss = args.best_loss

        for var in vars_to_replace:
            setattr(args, var, store_dict[var])
    else:
        curr_epoch = 0
        best_loss = np.inf
        model_dict, optim_dict = None, None

    for split in ['train', 'val']:
        loader, dataset = get_loader(args.root, args.batch_size, args.resize,
                                     args.imsize,
                                     augment=True,
                                     split=split, mode=split,
                                     text_only_data=False)
        loaders[split] = loader

    # create dataloader for training samples without images
    use_extra_data = True
    if args.recipe_loss_weight > 0 and use_extra_data:
        loader_textonly, _ = get_loader(args.root, args.batch_size*2,
                                        args.resize,
                                        args.imsize,
                                        augment=True,
                                        split='train', mode='train',
                                        text_only_data=True)

    vocab_size = len(dataset.get_vocab())
    model = get_model(args, vocab_size)

    params_backbone = list(model.image_encoder.backbone.parameters())
    params_fc = list(model.image_encoder.fc.parameters()) \
                + list(model.text_encoder.parameters()) \
                + list(model.merger_recipe.parameters()) \
                + list(model.projector_recipes.parameters())

    print("recipe encoder", count_parameters(model.text_encoder))
    print("image encoder", count_parameters(model.image_encoder))

    optimizer = get_optimizer(params_fc,
                              params_backbone,
                              args.lr, args.scale_lr, args.wd,
                              freeze_backbone=args.freeze_backbone)

    if model_dict is not None:
        model.load_state_dict(model_dict)
        if args.load_optimizer:
            try:
                optimizer.load_state_dict(optim_dict)
            except:
                print("Could not load optimizer state. Using default initialization...")

    ngpus = 1
    if device != 'cpu' and torch.cuda.device_count() > 1:
        ngpus = torch.cuda.device_count()
        model = nn.DataParallel(model)

    model = model.to(device)

    if device != 'cpu':
        cudnn.benchmark = True

    # learning rate scheduler
    scheduler = get_scheduler(args, optimizer)

    loss_function = TripletLoss(margin=args.margin)
    # training loop
    wait = 0

    scaler = GradScaler()

    for epoch in range(curr_epoch, args.n_epochs):

        for split in ['train', 'val']:
            if split == 'train':
                model.train()
            else:
                model.eval()

            metrics_epoch = defaultdict(list)

            total_step = len(loaders[split])
            loader = iter(loaders[split])

            if args.recipe_loss_weight > 0 and use_extra_data:
                iterator_textonly = iter(loader_textonly)

            img_feats, recipe_feats = None, None

            emult = 2 if (args.recipe_loss_weight > 0 and use_extra_data and split == 'train') else 1

            for i in range(total_step*emult):

                # sample from paired or text-only data loaders - only do this for training
                if i%2 == 0 and emult == 2:
                    this_iter_loader = iterator_textonly
                else:
                    this_iter_loader = loader

                optimizer.zero_grad()
                model.zero_grad()
                img_feat, recipe_feat, metrics_epoch = trainIter(args, split,
                                                                 this_iter_loader,
                                                                 model, optimizer,
                                                                 loss_function,
                                                                 scaler,
                                                                 metrics_epoch)

                if img_feat is not None:
                    if img_feats is not None:
                        img_feats = np.vstack((img_feats, img_feat))
                        recipe_feats = np.vstack((recipe_feats, recipe_feat))
                    else:
                        img_feats = img_feat
                        recipe_feats = recipe_feat

                if not args.tensorboard and i != 0 and i % args.log_every == 0:
                    # log metrics to stdout every few iterations
                    avg_metrics = {k: np.mean(v) for k, v in metrics_epoch.items() if v}
                    text_ = "split: {:s}, epoch [{:d}/{:d}], step [{:d}/{:d}]"
                    values = [split, epoch, args.n_epochs, i, total_step]
                    for k, v in avg_metrics.items():
                        text_ += ", " + k + ": {:.4f}"
                        values.append(v)
                    str_ = text_.format(*values)
                    print(str_)

            # computes retrieval metrics (average of 10 runs on 1k sized rankings)
            retrieval_metrics = computeAverageMetrics(img_feats, recipe_feats,
                                                      1000, 10, forceorder=True)

            for k, v in retrieval_metrics.items():
                metrics_epoch[k] = v

            avg_metrics = {k: np.mean(v) for k, v in metrics_epoch.items() if v}
            # log to stdout at the end of the epoch (for both train and val splits)
            if not args.tensorboard:
                text_ = "AVG. split: {:s}, epoch [{:d}/{:d}]"
                values = [split, epoch, args.n_epochs]
                for k, v in avg_metrics.items():
                    text_ += ", " + k + ": {:.4f}"
                    values.append(v)
                str_ = text_.format(*values)
                print(str_)

            # log to tensorboard at the end of one epoch
            if args.tensorboard:
                # 1. Log scalar values (scalar summary)
                for k, v in metrics_epoch.items():
                    logger.add_scalar('{}/{}'.format(split, k), np.mean(v), epoch)

        # monitor best loss value for early stopping
        # if the early stopping metric is recall (the higher the better),
        # multiply this value by -1 to save the model if the recall increases.
        if args.es_metric.startswith('recall'):
            mult = -1
        else:
            mult = 1

        curr_loss = np.mean(metrics_epoch[args.es_metric])

        if args.lr_decay_factor != -1:
            if args.scheduler_name == 'ReduceLROnPlateau':
                scheduler.step(curr_loss)
            else:
                scheduler.step()

        if curr_loss*mult < best_loss:
            if not args.tensorboard:
                print("Updating best checkpoint")
            save_model(model, optimizer, 'best', checkpoints_dir, ngpus)
            best_loss = curr_loss*mult

            wait = 0
        else:
            wait += 1

        # save current model state to be able to resume it
        save_model(model, optimizer, 'curr', checkpoints_dir, ngpus)
        args.best_loss = best_loss
        args.curr_epoch = epoch
        pickle.dump(args, open(os.path.join(checkpoints_dir,
                                            'args.pkl'), 'wb'))

        if wait == args.patience:
            break

    if args.tensorboard:
        logger.close()


def main():
    args = get_args()
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    random.seed(1234)
    np.random.seed(1234)
    train(args)


if __name__ == "__main__":
    main()
