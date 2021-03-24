# Revamping Cross-Modal Recipe Retrieval with Hierarchical Transformers and Self-supervised Learning

This is the PyTorch companion code for the paper:

*Amaia Salvador, Erhan Gundogdu, Loris Bazzani, and Michael Donoser. [Revamping Cross-Modal Recipe Retrieval with Hierarchical Transformers and Self-supervised Learning](https://www.amazon.science/publications/revamping-cross-modal-recipe-retrieval-with-hierarchical-transformers-and-self-supervised-learning). CVPR 2021*

If you find this code useful in your research, please consider citing using the following BibTeX entry:

```
@inproceedings{salvador2021revamping,
    title={Revamping Cross-Modal Recipe Retrieval with Hierarchical Transformers and Self-supervised Learning},
    author={Salvador, Amaia and Gundogdu, Erhan and Bazzani, Loris and Donoser, Michael},
    booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {June},
    year = {2021}
}
```

## Cloning

This repository uses [git-lfs](https://git-lfs.github.com/) to store model checkpoint files. Make sure to install it before cloning by following the instructions [here](https://github.com/git-lfs/git-lfs/wiki/Installation):

Once installed, model checkpoint files will be automatically downloaded when cloning the repository with:

```
git clone git@github.com:amzn/image-to-recipe-transformers.git
```

These files can optionally be ignored by using ```git lfs install --skip-smudge``` before cloning the repository, and can be downloaded at any time using ```git lfs pull```.

## Installation

- Create conda environment: ```conda env create -f environment.yml```
- Activate it with ```conda activate im2recipetransformers```

## Data preparation

- Download & uncompress Recipe1M [dataset](http://im2recipe.csail.mit.edu/dataset/download). The contents of the directory ```DATASET_PATH``` should be the following:

```
layer1.json
layer2.json
train/
val/
test/
```

The directories ```train/```, ```val/```, and ```test/``` must contain the image files for each split after uncompressing.

- Make splits and create vocabulary by running:

```
python preprocessing.py --root DATASET_PATH
```

This process will create auxiliary files under ```DATASET_PATH/traindata```, which will be used for training.

## Training

- Launch training with:

```
python train.py --model_name model --root DATASET_PATH --save_dir /path/to/saved/model/checkpoints
```

Tensorboard logging can be enabled with ```--tensorboard```. Then, from the checkpoints directory run:

```
tensorboard --logdir "./" --port PORT
```

Run ```python train.py --help``` for the full list of available arguments.


## Evaluation

- Extract features from the trained model for the test set samples of Recipe1M:

```
python test.py --model_name model --eval_split test --root DATASET_PATH --save_dir /path/to/saved/model/checkpoints
```

- Compute MedR and recall metrics for the extracted feature set:

```
python eval.py --embeddings_file /path/to/saved/model/checkpoints/model/feats_test.pkl --medr_N 10000
```

## Pretrained models

- We provide pretrained model weights under the ```checkpoints``` directory. Make sure you run ```git lfs pull``` to download the model files.
- Extract the zip files. For each model, a folder named ```MODEL_NAME``` with two files, ```args.pkl```, and ```model-best.ckpt``` is provided.
- Extract features for the test set samples of Recipe1M using one of the pretrained models by running:

```
python test.py --model_name MODEL_NAME --eval_split test --root DATASET_PATH --save_dir ../checkpoints
```

- A file with extracted features will be saved under ```../checkpoints/MODEL_NAME```.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.
