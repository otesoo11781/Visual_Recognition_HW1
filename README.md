# Visual_Recognition_HW1
This is howework 1 for selected topics in visual recongnition using deep learning. The goal is to classify the car brand of given images.

I use TResNet, which is a SOTA model on ImageNet classificaition task, to achieve 93% top-1 accuracy on the car brand classification.

The key idea is to transfer the TResNet model pretrained on ImageNet to the specific fine-grained dataset (car brand dataset provided by TA).

The source code is highly borrowed from [TResNet](https://github.com/mrT23/TResNet) and [Pytorch Tutorials](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)

## Hardware
The following specs were used to create the original solution:
- Ubuntu 16.04 LTS
- RTX 2080 with CUDA=10.1

## Reproducing Submission
To reproduct my submission without retrainig, do the following steps:
1. [Installation](#installation)
2. [Dataset Preparation](#dataset-preparation)
3. [Transfer Training](#transfer-training)
4. [Inference](#inference)
5. [Make Submission](#make-submission)

## Installation
All requirements should be detailed in requirements.txt. Using Anaconda is strongly recommended.
```
conda create -n [env_name] python=3.6
conda activate [env_name]
pip install -r requirements.txt
```

If error occurs during installation, please manually install them (refer to: [Pytorch](https://pytorch.org/get-started/locally/) and [Inplace_abn](https://github.com/mapillary/inplace_abn))

Note: Inplace_abn only supports Linux and CUDA version>=10.1

## Dataset Preparation
Download the dataset from the [kaggle website](https://www.kaggle.com/c/cs-t0828-2020-hw1/data)

Then, unzip them and put it under the **./data/** directory

Hence, the data directory is structured as:
```
./data
  +- training_data
  |  +- training_data
     |  +- 00001.jpg ...
  +- testing_data
  |  +- testing_data
     |  +- 00001.jpg ...
  +- training_labels.csv
```

## Transfer Training
### Retrain the model which pretrained on ImageNet
If you don't want to retrain the model, you can skip this step and download the trained weights on [Pretrained models](#pretrained-models)

Now, let's transfer train the model:

1. you should download the ImageNet pretrained TResNet model (e.g. tresnet_xl_448.pth) from [TResNet Model Zoo](https://github.com/mrT23/TResNet/blob/master/MODEL_ZOO.md).

2. put the model in the **./checkpoints/** directory.

3. run the following command:
```
$ python transfer.py --dataset_path=./data/training_data/training_data --label_path=./data/training_labels.csv --model_path=./checkpoints/tresnet_xl_448.pth --model_name=tresnet_xl --epochs=100 --batch_size=12
```
It takes about 17 hours to train the model and outputs 2 files:
1. **./checkpoints/transfer_model.pth**: the trained weights on given car brand dataset
2. **./class_name.npy**: the class names of car brand dataset

### Pretrained models
You can download pretrained model (**transfer_model.pth**) that used for my submission from [Here](https://drive.google.com/drive/folders/1Hj7sXE6OJt12IlDH7sQKuFU1l5dBZl1-?usp=sharing).

Then, put it under the **./checkpoints/** directory:
```
./checkpoints
  +- tresnet_xl_448.pth
  +- transfer_model.pth
```

In addition, the **class_name.npy** has been contained in this repo.

## Inference
With the testing dataset and trained model, you can run the following command to obtain the prediction results:
```
$ python test.py --test_dir=./data/testing_data/testing_data --model_path=./checkpoints/transfer_model.pth --model_name=tresnet_xl
```
After that, you will get classification results (**./result.csv**) which statisfy the submission format.

Note: please ensure **./class_name.npy** exists.

## Make Submission
Please go to [submission page](https://www.kaggle.com/c/cs-t0828-2020-hw1/submit) of kaggle website and upload the **result.csv** obtained in the previous step.

Note: the repo has provided **./result.csv** which is corresponding to my submission on leaderboard with accuracy 0.9304

