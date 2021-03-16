# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from torchvision.models import resnet18, resnet50, resnet101, resnet152, inception_v3, resnext50_32x4d, resnext101_32x8d
import timm
import torch
import torch.nn as nn
import random
import numpy as np
import torch.nn.functional as F
import math


class LearnedPositionalEncoding(nn.Module):
    """ Positional encoding layer

    Parameters
    ----------
    dropout : float
        Dropout value.
    num_embeddings : int
        Number of embeddings to train.
    hidden_dim : int
        Embedding dimensionality
    """

    def __init__(self, dropout=0.1, num_embeddings=50, hidden_dim=512):
        super(LearnedPositionalEncoding, self).__init__()

        self.weight = nn.Parameter(torch.Tensor(num_embeddings, hidden_dim))
        self.dropout = nn.Dropout(p=dropout)
        self.hidden_dim = hidden_dim

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.weight)

    def forward(self, x):
        batch_size, seq_len = x.size()[:2]
        embeddings = self.weight[:seq_len, :].view(1, seq_len, self.hidden_dim)
        x = x + embeddings
        return self.dropout(x)


def AvgPoolSequence(attn_mask, feats, e=1e-12):
    """ The function will average pool the input features 'feats' in
        the second to rightmost dimension, taking into account
        the provided mask 'attn_mask'.
    Inputs:
        attn_mask (torch.Tensor): [batch_size, ...x(N), 1] Mask indicating
                                  relevant (1) and padded (0) positions.
        feats (torch.Tensor): [batch_size, ...x(N), D] Input features.
    Outputs:
        feats (torch.Tensor) [batch_size, ...x(N-1), D] Output features
    """

    length = attn_mask.sum(-1)
    # pool by word to get embeddings for a sequence of words
    mask_words = attn_mask.float()*(1/(length.float().unsqueeze(-1).expand_as(attn_mask) + e))
    feats = feats*mask_words.unsqueeze(-1).expand_as(feats)
    feats = feats.sum(dim=-2)

    return feats


class ViTBackbone(nn.Module):
    """Class for ViT models

    Parameters
    ----------
    hidden_size : int
        Embedding size.
    image_model : string
        Model name to load.
    pretrained : bool
        Whether to load pretrained imagenet weights.

    """
    def __init__(self, hidden_size, image_model,
                 pretrained=True):
        super(ViTBackbone, self).__init__()

        self.backbone = timm.create_model(image_model, pretrained=True)
        in_feats = self.backbone.head.in_features
        self.fc = nn.Linear(in_feats, hidden_size)

    def forward(self, images, freeze_backbone=False):
        if not freeze_backbone:
            feats = self.backbone.forward_features(images)
        else:
            with torch.no_grad():
                feats = self.backbone.forward_features(images)
        out = self.fc(feats)
        return nn.Tanh()(out)


class TorchVisionBackbone(nn.Module):
    """Class for Torchvision models

    Parameters
    ----------
    hidden_size : int
        Embedding size.
    image_model : string
        Model name to load.
    pretrained : bool
        Whether to load pretrained imagenet weights.

    """
    def __init__(self, hidden_size, image_model, pretrained=True):
        super(TorchVisionBackbone, self).__init__()
        self.image_model = image_model
        backbone = globals()[image_model](pretrained=pretrained)
        modules = list(backbone.children())[:-2]
        self.backbone = nn.Sequential(*modules)
        in_feats = backbone.fc.in_features

        self.fc = nn.Linear(in_feats, hidden_size)

    def forward(self, images, freeze_backbone=False):
        """Extract feature vectors from input images."""
        if not freeze_backbone:
            feats = self.backbone(images)
        else:
            with torch.no_grad():
                feats = self.backbone(images)
        feats = feats.view(feats.size(0), feats.size(1),
                           feats.size(2)*feats.size(3))

        feats = torch.mean(feats, dim=-1)
        out = self.fc(feats)

        return nn.Tanh()(out)


class SingleTransformerEncoder(nn.Module):
    """A transformer encoder with masked average pooling at the output

    Parameters
    ----------
    dim : int
        Embedding dimensionality.
    n_heads : int
        Number of attention heads.
    n_layers : int
        Number of transformer layers.

    """
    def __init__(self, dim, n_heads, n_layers):
        super(SingleTransformerEncoder, self).__init__()

        self.pos_encoder = LearnedPositionalEncoding(hidden_dim=dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=dim,
                                                   nhead=n_heads)

        self.tf = nn.TransformerEncoder(encoder_layer,
                                        num_layers=n_layers)

    def forward(self, feat, ignore_mask):

        if self.pos_encoder is not None:
            feat = self.pos_encoder(feat)
        # reshape input to t x bs x d
        feat = feat.permute(1, 0, 2)
        out = self.tf(feat, src_key_padding_mask=ignore_mask)
        # reshape back to bs x t x d
        out = out.permute(1, 0, 2)

        out = AvgPoolSequence(torch.logical_not(ignore_mask), out)

        return out


class RecipeTransformerEncoder(nn.Module):
    """The recipe text encoder. Encapsulates encoders for all recipe components.

    Parameters
    ----------
    vocab_size : int
        Input size (recipe vocabulary).
    hidden_size : int
        Output embedding size.
    n_heads : int
        Number of attention heads.
    n_layers : int
        Number of transformer layers.

    """
    def __init__(self, vocab_size, hidden_size, n_heads,
                 n_layers):
        super(RecipeTransformerEncoder, self).__init__()

        # Embedding layer mapping vocabulary words to embeddings.
        # The embedding layer is common for all text tokens (ingrs, instrs, title)
        self.word_embedding = nn.Embedding(vocab_size, hidden_size)

        self.tfs = nn.ModuleDict()

        # independent transformer encoder for each recipe component
        for name in ['title', 'ingredients', 'instructions']:
            self.tfs[name] = SingleTransformerEncoder(dim=hidden_size,
                                                      n_heads=n_heads,
                                                      n_layers=n_layers
                                                     )



        # second transformer for sequences of sequences inputs
        # (eg a list of raw ingredients or a list of instructions)
        self.merger = nn.ModuleDict()
        for name in ['ingredients', 'instructions']:
            self.merger[name] = SingleTransformerEncoder(dim=hidden_size,
                                                         n_heads=n_heads,
                                                         n_layers=n_layers)

    def forward(self, input, name=None):
        '''
        Extracts features for an input using the corresponding encoder (by name)
        '''
        # check if input is a sequence or a sequence of sequences
        if len(input.size()) == 2:
            # if it is a sequence, the output of a single transformer is used
            ignore_mask = (input == 0)
            out = self.tfs[name](self.word_embedding(input), ignore_mask)

        else:
            # if it's a sequence of sequences, the first encoder is applied
            # to each sentence, and the second on

            # reshape from BxNxTxD to BNxTxD
            input_rs = input.view(input.size(0)*input.size(1), input.size(2))
            ignore_mask = (input_rs == 0)

            # trick to avoid nan behavior with fully padded sentences
            # (due to batching)
            ignore_mask[:, 0] = 0
            out = self.tfs[name](self.word_embedding(input_rs), ignore_mask)

            # reshape back
            out = out.view(input.size(0), input.size(1), out.size(-1))

            # create mask for second transformer
            attn_mask = input > 0
            mask_list = (attn_mask.sum(dim=-1) > 0).bool()

            out = self.merger[name](out, torch.logical_not(mask_list))

        return out


def get_image_model(hidden_size, image_model, pretrained=True):
    if 'vit' in image_model:
        image_model = ViTBackbone(hidden_size, image_model, pretrained)
    else:
        image_model = TorchVisionBackbone(hidden_size, image_model, pretrained)

    return image_model


class JointEmbedding(nn.Module):
    """A joint embedding of ingredients and recipes
    Parameters
    ----------
    output_size : int
        Embedding output size.
    image_model : string
        Name of image model.
    vocab_size : int
        Input size for recipes.
    hidden_recipe : int
        Embedding size for recipe components
    n_heads : int
        Number of attention heads.
    n_layers : int
        Number of transformer layers.
    """

    def __init__(self, output_size, image_model='resnet18',
                 vocab_size=None,
                 hidden_recipe=512,
                 n_heads=4, n_layers=2):
        super(JointEmbedding, self).__init__()


        self.text_encoder = RecipeTransformerEncoder(vocab_size,
                                                     hidden_size=hidden_recipe,
                                                     n_heads=n_heads,
                                                     n_layers=n_layers)

        self.image_encoder = get_image_model(hidden_size=output_size,
                                             image_model=image_model)
        self.merger_recipe = nn.ModuleList()

        # linear layer to merge features from all recipe components.
        self.merger_recipe = nn.Linear(hidden_recipe*(3), output_size)

        # projection layers for self supervised recipe loss
        self.projector_recipes = nn.ModuleDict()
        names = ['title', 'ingredients', 'instructions']
        for name in names:
            self.projector_recipes[name] = nn.ModuleDict()
            for name2 in names:
                if name2 != name:
                    self.projector_recipes[name][name2] = nn.Linear(hidden_recipe, hidden_recipe)

    def forward(self, img, title, ingrs, instrs,
                freeze_backbone=True):

        text_features = []
        projected_text_features = {'title': {},
                                   'ingredients': {},
                                   'instructions': {},
                                   'raw': {}}

        elems = {'title': title, 'ingredients': ingrs, 'instructions': instrs}

        names = list(elems.keys())

        for name in names:
            # for each recipe component, extracts features and projects them to all other spaces
            input_source = elems[name]
            text_feature = self.text_encoder(input_source, name)
            text_features.append(text_feature)
            projected_text_features['raw'][name] = text_feature
            for name2 in names:
                if name2 != name:
                    projected_text_features[name][name2] = self.projector_recipes[name][name2](text_feature)

        if img is not None:
            img_feat = self.image_encoder(img, freeze_backbone=freeze_backbone)
        else:
            img_feat = None


        recipe_feat = self.merger_recipe(torch.cat(text_features, dim=1))

        recipe_feat = nn.Tanh()(recipe_feat)

        return img_feat, recipe_feat, projected_text_features


def get_model(args, vocab_size):
    model = JointEmbedding(vocab_size=vocab_size,
                           output_size=args.output_size,
                           hidden_recipe=args.hidden_recipe,
                           image_model=args.backbone,
                           n_heads=args.tf_n_heads,
                           n_layers=args.tf_n_layers,
                           )
    return model
