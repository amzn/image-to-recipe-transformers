# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import multiprocessing
from dataset import get_loader
from models import get_model
import torch.backends.cudnn as cudnn
from config import get_args
from tqdm import tqdm
import torch
import numpy as np
import pickle
from utils.utils import load_checkpoint, count_parameters

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
map_loc = None if torch.cuda.is_available() else 'cpu'


def test(args):

    if device != 'cpu':
        cudnn.benchmark = True
    checkpoints_dir = os.path.join(args.save_dir, args.model_name)
    # make sure these arguments are kept from commandline and not from loaded args
    vars_to_replace = ['batch_size', 'eval_split', 'imsize', 'root', 'save_dir']
    store_dict = {}
    for var in vars_to_replace:
        store_dict[var] = getattr(args, var)
    args, model_dict, _ = load_checkpoint(checkpoints_dir, 'best', map_loc,
                                          store_dict)
    for var in vars_to_replace:
        setattr(args, var, store_dict[var])

    loader, dataset = get_loader(args.root, args.batch_size, args.resize,
                                 args.imsize,
                                 augment=False,
                                 split=args.eval_split, mode='test',
                                 drop_last=False)
    print("Extracting features for %d samples from the %s set..."%(len(dataset),
                                                                   args.eval_split))
    vocab_size = len(dataset.get_vocab())
    model = get_model(args, vocab_size)

    print("recipe encoder", count_parameters(model.text_encoder))
    print("image encoder", count_parameters(model.image_encoder))


    model.load_state_dict(model_dict, strict=False)

    if device != 'cpu' and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    
    model = model.to(device)
    model.eval()

    total_step = len(loader)
    loader = iter(loader)
    print("Loaded model from %s ..."%(checkpoints_dir))
    all_f1, all_f2 = None, None
    allids = []

    for _ in tqdm(range(total_step)):

        img, title, ingrs, instrs, ids = loader.next()

        img = img.to(device)
        title = title.to(device)
        ingrs = ingrs.to(device)
        instrs = instrs.to(device)
        with torch.no_grad():
            out = model(img, title, ingrs, instrs)
            f1, f2, _ = out
        allids.extend(ids)
        if all_f1 is not None:
            all_f1 = np.vstack((all_f1, f1.cpu().detach().numpy()))
            all_f2 = np.vstack((all_f2, f2.cpu().detach().numpy()))
        else:
            all_f1 = f1.cpu().detach().numpy()
            all_f2 = f2.cpu().detach().numpy()

    print("Done.")

    file_to_save = os.path.join(checkpoints_dir,
                                'feats_' + args.eval_split+'.pkl')
    print(np.shape(all_f1))

    with open(file_to_save, 'wb') as f:
        pickle.dump(all_f1, f)
        pickle.dump(all_f2, f)
        pickle.dump(allids, f)
    print("Saved features to disk.")


if __name__ == "__main__":
    args = get_args()
    test(args)
