# 20201021 
"""
Training data: wnut17train.conll (Twitter)
Development data: emerging.dev.conll (YouTube)
Test data with tags: emerging.test.annotated

--Test data (no tags): emerging.test (StackExchange and Reddit)--
"""

train_file = 'wnut17train.conll'
dev_file = 'emerging.dev.conll'
test_file = 'emerging.test.annotated'
import pdb 
import os 
from pathlib import Path
import random 

def getDataLen(train_file, dev_file, test_file):
  save_dir = 'data/'
  if not os.path.exists(save_dir):
    os.mkdir(save_dir)
  save_dir = Path(save_dir)

  with open(train_file, 'r') as f1:
    train_data = list(filter(None, f1.read().split('\t\n')))
  print('Train data len: {}'.format(len(train_data)))
  with open(save_dir / 'train.txt', 'w') as f2:
    for trd in train_data:
      trd.replace('\t', ' ')
      f2.write(trd)
      f2.write('\n')
  print('Train data save successfully.')
    
  with open(dev_file, 'r') as f1:
    dev_data = list(filter(None, f1.read().split('\n\n')))
  print('Dev data len: {}'.format(len(dev_data)))
  with open(save_dir / 'dev.txt', 'w') as f2:
    for dd in dev_data:
      dd.replace('\t', ' ')
      f2.write(dd)
      f2.write('\n\n')
  print('Dev data save successfully.')

  with open(test_file, 'r') as f1:
    test_data = list(filter(None, f1.read().split('\n\n')))
  print('Test data len: {}'.format(len(test_data)))
  with open(save_dir / 'test.txt', 'w') as f2:
    for td in test_data:
      td.replace('\t', ' ')
      f2.write(td)
      f2.write('\n\n')
  print('Test data save successfully.')
  

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
    print('Test data length is {}'.format(len(test_data)))
  if sample_num:
    sampled_train_data = random.sample(train_data, sample_num)
  else:
    sampled_train_data = train_data
  print('After sampling, train data length is {}'.format(len(sampled_train_data)))

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
  # getDataLen(train_file, dev_file, test_file)
  # Train data len: 2395  |  Dev data len: 1009  |  Test data len: 1287

  sampleData('data/train.txt', 'data/test.txt', '1500_data', 1500)
  sampleData('1500_data/train.txt', 'data/test.txt', '150_data', 150)
  sampleData('1500_data/train.txt', 'data/test.txt', '15_data', 15)

