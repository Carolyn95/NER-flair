# 20201020 
from nemo.collections import nlp as nemo_nlp
from nemo.utils.exp_manager import exp_manager

import os
import wget 
import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf

# DATA_DIR = "DATA_DIR"
# WORK_DIR = "WORK_DIR"
# MODEL_CONFIG = "token_classification_config.yaml"

## Download preprocessed GMB data
# os.makedirs(WORK_DIR, exist_ok=True)
# os.makedirs(DATA_DIR, exist_ok=True)
# print('Downloading GMB data...')
# wget.download('https://dldata-public.s3.us-east-2.amazonaws.com/gmb_v_2.2.0_clean.zip', DATA_DIR)

## Download config file
# config_dir = WORK_DIR + '/configs/'
# os.makedirs(config_dir, exist_ok=True)
# if not os.path.exists(config_dir + MODEL_CONFIG):
#     print('Downloading config file...')
#     wget.download('https://raw.githubusercontent.com/NVIDIA/NeMo/main/examples/nlp/token_classification/conf/' + MODEL_CONFIG, config_dir)
# else:
#     print ('config file is already exists')

import pdb
import os 
from pathlib import Path
import numpy as np 
import random 
from collections import Counter

seed = 2020
random.seed(seed)
np.random.seed(seed)


def GmbToConllFormat(text_file, label_file, save_dir):
  with open(text_file, 'r') as f:
    text_data = list(filter(None, f.read().split('\n')))
  with open(label_file, 'r') as f:
    label_data = list(filter(None, f.read().split('\n')))
  
  new_data_file = 'new_' + os.path.split(text_file)[-1].split('_')[-1]

  with open(save_dir + new_data_file, 'w') as f:
    for text, label in zip(text_data, label_data):
      text_list = text.split()
      label_list = label.split()
      for t, l in zip(text_list, label_list):
        f.write(t + ' ' + l + '\n')
      f.write('\n')
  print('Writing  to {} completed!'.format(save_dir + new_data_file))

 
def sampleData(train_file, test_file, save_dir, sample_num=None):
  if not os.path.exists(save_dir):
    os.mkdir(save_dir)

  save_dir = Path(save_dir)
  with open(train_file, 'r') as f:
    train_data = list(filter(None, f.read().split('\n\n')))
    random.shuffle(train_data)
    print('Train data length is {}'.format(len(train_data)))
  with open(test_file, 'r') as f:
    test_data = list(filter(None, f.read().split('\n\n')))
    if 'full' in str(save_dir):
      random.shuffle(test_data)
    print('Test data length is {}'.format(len(test_data)))
  if sample_num:
    sampled_train_data = random.sample(train_data, sample_num)
  else:
    sampled_train_data = train_data

  # save txt for fitting pipeline data format
  with open(save_dir / 'train.txt', 'w') as f:
    for std in sampled_train_data:
      f.write(std)
      f.write('\n\n')
  with open(save_dir / 'test.txt', 'w') as f:
    for td in test_data:
      f.write(td)
      f.write('\n\n')



if __name__ == '__main__':
  print()
  GmbToConllFormat('data/text_dev.txt', 'data/labels_dev.txt', 'processed_data/')
  GmbToConllFormat('data/text_train.txt', 'data/labels_train.txt', 'processed_data/')
  sampleData('processed_data/new_train.txt', 'processed_data/new_dev.txt', 'processed_data/full_data', None)
  sampleData('processed_data/full_data/train.txt', 'processed_data/full_data/test.txt', 'processed_data/1500_data', 1500)
  sampleData('processed_data/1500_data/train.txt', 'processed_data/full_data/test.txt', 'processed_data/150_data', 150)
  sampleData('processed_data/1500_data/train.txt', 'processed_data/full_data/test.txt', 'processed_data/15_data', 15)

  
