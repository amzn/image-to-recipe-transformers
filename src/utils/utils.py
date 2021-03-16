# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

""" General project utilities
"""
import os
import torch
import nltk
import pickle

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_scheduler(args, optimizer):

    if args.scheduler_name == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=args.lr_decay_patience,
                                                    gamma=args.lr_decay_factor)

    elif args.scheduler_name == 'ReduceLROnPlateau':
        mode = 'max' if 'recall' in args.es_metric else 'min'
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               mode=mode,
                                                               factor=args.lr_decay_factor,
                                                               patience=args.lr_decay_patience)
    else:

        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,
                                                           gamma=args.lr_decay_factor)


    return scheduler


def get_optimizer(params_fc,
                  params_backbone,
                  lr, scale_lr, wd,
                  freeze_backbone):

    if freeze_backbone:
        optimizer = torch.optim.Adam(params_fc, lr=lr, weight_decay=wd)
    else:
        optimizer = torch.optim.Adam([{'params': params_fc},
                                      {'params': params_backbone,
                                       'lr': lr*scale_lr}],
                                     lr=lr, weight_decay=wd)

    return optimizer


def make_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)


def get_token_ids(sentence, vocab):
    """
    get vocabulary tokens for each word in a sentence
    """
    tok_ids = []
    tokens = nltk.tokenize.word_tokenize(sentence.lower())

    tok_ids.append(vocab['<start>'])
    for token in tokens:
        if token in vocab:
            tok_ids.append(vocab[token])
        else:
            # unk words will be ignored
            tok_ids.append(vocab['<unk>'])
    tok_ids.append(vocab['<end>'])
    return tok_ids


def save_model(model, optim, suffix, checkpoints_dir, ngpus):
    if ngpus > 1:
        model_dict = model.module.state_dict()
    else:
        model_dict = model.state_dict()
    torch.save(model_dict, os.path.join(checkpoints_dir,
                                        'model-'+suffix+'.ckpt'))
    torch.save(optim.state_dict(), os.path.join(checkpoints_dir,
                                                'optim-'+suffix+'.ckpt'))


def load_checkpoint(checkpoints_dir, suff, map_loc, vars_to_replace):

    assert os.path.exists(os.path.join(checkpoints_dir, 'args.pkl'))
    model_state_dict = torch.load(os.path.join(checkpoints_dir,
                                               'model-'+suff+'.ckpt'),
                                  map_location=map_loc)
    # optimizer can be skipped 
    try:
        opt_state_dict = torch.load(os.path.join(checkpoints_dir,
                                                 'optim-'+suff+'.ckpt'),
                                    map_location=map_loc)
    except:
        opt_state_dict = None

    args = pickle.load(open(os.path.join(checkpoints_dir, 'args.pkl'), 'rb'))
    for var in vars_to_replace:
        setattr(args, var, vars_to_replace[var])

    return args, model_state_dict, opt_state_dict


def list2Tensors(input_list):
    """Given a list of lists of variable-length elements, return a 2D tensor padded with 0s
    """
    max_seq_len = max(map(len, input_list))
    output = [v + [0] * (max_seq_len - len(v)) for v in input_list]

    return torch.Tensor(output)
