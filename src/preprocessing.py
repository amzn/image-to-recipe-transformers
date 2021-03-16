# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# -*- coding: utf-8 -*-
"""
Preprocessing script
"""
import os
from collections import Counter
import pickle
import json
from tqdm import tqdm
import nltk
from config import get_preprocessing_args
from utils.utils import make_dir


def run(root, min_freq=10, force=False):
    """Main preprocessing function.

    Parameters
    ----------
    root : string
        Path to Recipe1M dataset.
    min_freq : int
        Minimum number of occurrences to keep a word in the dictionary.
    force : bool
        Force re-computation of word counter.
    """

    # fixing incorrect parsing
    replace_dict_rawingrs = {'1/2 ': ['12 '], '1/3 ': ['13 '], '1/4 ': ['14 '],
                             '1/8 ': ['18 '],
                             '2/3 ': ['23 '], '3/4 ': ['34 ']
                             }
    folder_name = 'traindata'
    save_dir = os.path.join(root, folder_name)
    make_dir(save_dir)

    layers = {}

    datasets = {'train': {}, 'val': {}, 'test': {}}
    datasets_noimages = {'train': {}, 'val': {}, 'test': {}}
    layers_to_load = ['layer1', 'layer2']

    # load recipe1m layers
    for layer_name in layers_to_load:
        layer_file = os.path.join(root, layer_name+ '.json')
        with open(layer_file, 'r', encoding='utf-8') as data_file:
            layers[layer_name] = json.load(data_file)

    # load split files
    for split in ['train', 'val', 'test']:
        with open(os.path.join('../data/', split + '.txt'), 'r') as f:
            ids = f.readlines()
            print(len(ids))
            for item in ids:
                datasets[split][item.strip()] = {}

    # lookup tables for images
    id2im = {}
    for i, entry in enumerate(layers['layer2']):
        id2im[entry['id']] = i


    create_vocab = not os.path.exists('../data/vocab.pkl') or force
    if create_vocab:
        print("Creating vocabulary...")
        # initialize or load the word counter
        if not os.path.exists(os.path.join(save_dir, 'counter.pkl')) or force:
            counter = Counter()
            update_counter = True
        else:
            counter = pickle.load(open(os.path.join(save_dir, 'counter.pkl'), 'rb'))
            update_counter = False
    else:
        print("Found a vocabulary file. Skipping vocabulary creation.")
        update_counter = False

    print("Processing dataset...")
    for i, entry in tqdm(enumerate(layers['layer1'])):
        partition = entry['partition']
        ingrs_raw = [ingr['text'].lower() for ingr in entry['ingredients']]
        instrs = [instr['text'].lower() for instr in entry['instructions']]
        title = entry['title'].lower()

        # if they exist, copy image paths for this recipe
        if entry['id'] in datasets[partition].keys():
            ims = layers['layer2'][id2im[entry['id']]]
            images_list = [im['id'] for im in ims['images']]
        else:
            images_list = []

        # fix quantities in raw ingredients
        ingrs_clean = []
        for ingr in ingrs_raw:
            for k, v in replace_dict_rawingrs.items():
                for el in v:
                    if el in ingr:
                        ingr = ingr.replace(el, k)
            ingrs_clean.append(ingr)

        # counter is updated only for samples in the original
        # training set containing paired data only
        if partition == 'train' and update_counter and create_vocab and entry['id'] in datasets[partition].keys():
            title_words = nltk.tokenize.word_tokenize(title)
            counter.update(title_words)

            for _, text_elements in {'ingrs': ingrs_clean,
                                     'instrs': instrs}.items():
                for el in text_elements:
                    counter.update(nltk.tokenize.word_tokenize(el))

        new_entry = {}
        new_entry['ingredients'] = ingrs_clean
        new_entry['title'] = title
        new_entry['instructions'] = instrs
        new_entry['images'] = images_list

        # add the sample to the right dataset (paired/text-only)
        if entry['id'] in datasets[partition].keys():
            datasets[partition][entry['id']] = new_entry
        else:
            datasets_noimages[partition][entry['id']] = new_entry

    if update_counter and create_vocab:
        pickle.dump(counter, open(os.path.join(save_dir, 'counter.pkl'), 'wb'))

    if create_vocab:
        # create vocabulary
        vocab = {0: '<pad>', 1: '<start>', 2: '<end>'}
        print("Found %d unique words"%(len(counter)))
        words = [word for word, cnt in counter.items() if cnt >= min_freq]
        print("Removing words that appear less than %d times"%(min_freq))
        print("Number of remaining words: %d"%(len(words)))

        for i, word in enumerate(words):
            vocab[i+3] = word
        vocab[i+4] = '<unk>'

        # save vocabulary and dataset files by split
        pickle.dump(vocab, open(os.path.join('../data', 'vocab.pkl'), 'wb'))

    for key, data in datasets.items():
        pickle.dump(data, open(os.path.join(save_dir, key+'.pkl'), 'wb'))
        print(key, len(data))
    for key, data in datasets_noimages.items():
        pickle.dump(data, open(os.path.join(save_dir, key+'_noimages.pkl'), 'wb'))
        print(key, len(data))


if __name__ == "__main__":
    args = get_preprocessing_args()
    run(args.root, args.min_freq, args.force)
