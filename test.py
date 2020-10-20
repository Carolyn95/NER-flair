import pdb 
import pandas as pd 
from pathlib import Path 
import os 
import numpy as np 
from collections import Counter
import random

# add on 20201019, test using original model but swap testing data set
# environment: source ~/Projects/othersgit/BERT-NER/envr/bin/activate
# transformers version == 3.0.2 should work

import torch
seed = 2020
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

from flair.data import Corpus, Sentence
from flair.embeddings import TransformerWordEmbeddings, StackedEmbeddings, WordEmbeddings
from typing import List
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from flair.datasets import ColumnCorpus
import argparse
import pdb 
import traceback

def convertData(data_file_path, save_dir):
  if not os.path.exists(save_dir):
    os.mkdir(save_dir)
  save_dir = Path(save_dir)
  data = np.load(data_file_path, allow_pickle=True)
  with open(save_dir / 'test.txt', 'w') as f:
    # f.write('-DOCSTART- -X- -X- O\n\n')
    for d in data:
      f.write(d)
      f.write('\n\n')      
  print('Convertion done!')

# convertData('SanityTestData/all_test.npy', 'SanityTestData')

def testModel(model_dir, test_sent=None, test_file_dir=None):
  """
  model_dir: directory contains 'final_model.pt'
  test_sent: one sentence to test
  test_file: one file of sentences to test
  """
  if test_sent and test_file_dir:
    raise Exception("Argument conflicts, only one type of testing method is allowed.")
  elif not test_sent and not test_file_dir:
    raise Exception("Argument invalid, at least one testing method is required") 
  
  model_path = model_dir + '/final-model.pt'                            
  tagger = SequenceTagger.load(model_path)
  # pdb.set_trace()

  if test_sent:
    print('Predicting in singular mode')
    test_sent = Sentence(test_sent)
    try:
      tagger.predict(test_sent)
    except:
      traceback.print_exc()
    print(test_sent.to_tagged_string())

  if test_file_dir:
    print('Predicting in plural mode')
    try:
      columns = {0: 'text', 1: 'ner'}
      corpus = ColumnCorpus(test_file_dir, columns)
      test_data = corpus.test 
    except Exception:
      traceback.print_exc() 
      raise Exception('Directory must contain `test.txt` file, one column `text` the other `ner`')
    
    test_result, test_loss = tagger.evaluate(test_data, out_path=test_file_dir + '/test.tsv')
    result_line = f"\t{test_loss}\t{test_result.log_line}"
    print(f"TEST : loss {test_loss} - score {round(test_result.main_score, 4)}")
    print(f"TEST RESULT : {result_line}")

  print('Model prediction ends')


testModel('2pt5pct/models', test_file_dir='SanityTestData/data')
# testModel('20pct/models', test_sent='hello bert')
pdb.set_trace()


# add on 20201019, create synthetic test data, diff sentence structure from train, same set of entity
seed = 2020
random.seed(seed)
np.random.seed(seed)


def createSanityTestData(data_folder, file_name_list, save_path):
  if data_folder:
    file_name_list = [data_folder + fn for fn in file_name_list]
  if not os.path.exists(save_path):
    os.mkdir(save_path)
  save_path = Path(save_path)
  alldata = []
  for file_name in file_name_list:
    with open(file_name, 'r') as f:
      data = list(filter(None, f.read().split('\n\n')))
      alldata += data
  random.shuffle(alldata)
  alldata = np.array(alldata)
  np.save(save_path / 'all_test.npy', alldata)

data_folder = 'synthetic-data/'
file_name_list = ['new_applications.txt', 'new_devices.txt', 'new_locations.txt']
save_path = 'SanityTestData'

createSanityTestData(data_folder, file_name_list, save_path)


def strfSample(data_folder, is_randomized, sample_rate, save_path):
  """
  read a numpy array, count category in the array, save it in a list
  print the result
  stratified sampling from the newly created list 
  data_folder, save_path
  """
  def categorize(x):
    if 'APPLICATION' in x:
      return 'APPLICATION'
    elif 'DEVICE' in x:
      return 'DEVICE'
    elif 'LOCATION' in x:
      return 'LOCATION'
    elif 'TREE' in x:
      return 'TREE'
    else:
      raise EXCEPTION("Not recognisable entity")
  
  data_folder = Path(data_folder)
  save_path = Path(save_path)
  if not os.path.exists(save_path):
    os.mkdir(save_path)
  for filename in os.listdir(data_folder):
    if 'test' in filename:
      test_data = np.load(data_folder / filename, allow_pickle=True)
      np.save(save_path / 'test.npy', test_data)
    elif 'train' in filename:
      train_data = np.load(data_folder / filename, allow_pickle=True)
      train_cats = [categorize(x) for x in train_data]
      train_cats_counter = Counter(train_cats)
      cats = []
      samples = {}
      temp = []
      for i, c in enumerate(sorted(train_cats)):
        if c not in cats:
          cats.append(c)
          temp = []
        temp.append(i)      
        # pdb.set_trace()
        if temp and len(temp) == train_cats_counter[c]:
          sample_temp = random.sample(temp, int(len(temp) * sample_rate))
          samples[c] = sample_temp                  
    else:
      print('Not target file')
  return samples

test = strfSample('tmp', True, 1, 'test10pct')
pdb.set_trace()

# dummy-data-#/train.tmpl
serial_no = 1
data_dir = 'dummy-data/dummy-data-' + str(serial_no)
flag = 'train'

template_path = data_dir + flag + '.tmpl'
save_path = data_dir + flag + '.txt'

def makeData(template_path, save_path):
  with open(tmplate_path) as f:
    data = f.read()
    data = data.split('\n')
    data = [eval(d.replace("],", ']').replace('"', '')) for d in data]

  df = pd.DataFrame(data, columns=['text', 'annotation'])
  createData(df, save_path)

def makeData():
  """Show a smaill piece of example
  """
  data = pd.DataFrame(
      [
        ['Horses are too tall and they pretend to care about your feelings', [('Horses', 'ANIMAL')]],
        ['Who is Shaka Khan?', [('Shaka Khan', 'PERSON')]],
        ['I like London and Berlin.', [('London', 'LOCATION'), ('Berlin', 'LOCATION')]],
        ['There is a banyan tree in the courtyard', [('banyan tree', 'TREE')]],
        ['Dogs are more adorable than cats', [('Dogs', 'ANIMAL'), ('cats', 'ANIMAL')]], 
        ['John Watson is looking for his cap', [('John Watson', 'PERSON')]], 
        ['Beijing is the capital city of China', [('Beijing', 'LOCATION')]], 
        ['Leaves of Pine never yellow', [('Pine', 'TREE')]] 
      ], 
      columns=['text', 'annotation'])

  filepath_train = 'dummy-data/dummy-data-2/train.txt'
  filepath_dev = 'dummy-data/dummy-data-2/dev.txt'
  filepath_test = 'dummy-data/dummy-data-2/test.txt'
  createData(data, filepath_train)
  createData(data, filepath_dev)
  createData(data, filepath_test)

def makeTrainData(dummydata_tmpl):
  filepath_train = 'dummy-data/dummy-data-2/train.txt'
  data = pd.DataFrame(
    [
      ['Horses are too tall and they pretend to care about your feelings', [('Horses', 'ANIMAL')]],
      ['Who is Shaka Khan?', [('Shaka Khan', 'PERSON')]],
      ['I like London and Berlin.', [('London', 'LOCATION'), ('Berlin', 'LOCATION')]],
      ['There is a banyan tree in the courtyard', [('banyan tree', 'TREE')]],
      ['Dogs are more adorable than cats', [('Dogs', 'ANIMAL'), ('cats', 'ANIMAL')]], 
      ['John Watson is looking for his cap', [('John Watson', 'PERSON')]], 
      ['Beijing is the capital city of China', [('Beijing', 'LOCATION')]], 
      ['Leaves of Pine never yellow', [('Pine', 'TREE')]] 
    ], 
    columns=['text', 'annotation'])
  createData(data, filepath_train)
  print()

def makeDevData(dummydata_tmpl):
  filepath_dev = 'dummy-data/dummy-data-2/dev.txt'
  data = pd.DataFrame(
    [
      ['Horses are too tall and they pretend to care about your feelings', [('Horses', 'ANIMAL')]],
      ['Who is Shaka Khan?', [('Shaka Khan', 'PERSON')]],
      ['I like London and Berlin.', [('London', 'LOCATION'), ('Berlin', 'LOCATION')]],
      ['There is a banyan tree in the courtyard', [('banyan tree', 'TREE')]],
      ['Dogs are more adorable than cats', [('Dogs', 'ANIMAL'), ('cats', 'ANIMAL')]], 
      ['John Watson is looking for his cap', [('John Watson', 'PERSON')]], 
      ['Beijing is the capital city of China', [('Beijing', 'LOCATION')]], 
      ['Leaves of Pine never yellow', [('Pine', 'TREE')]] 
    ], 
    columns=['text', 'annotation'])  
  createData(data, filepath_dev)
  print()

def makeTestData(dummydata_tmpl):
  filepath_test = 'dummy-data/dummy-data-2/test.txt'
  data = pd.DataFrame(
    [
      ['Horses are too tall and they pretend to care about your feelings', [('Horses', 'ANIMAL')]],
      ['Who is Shaka Khan?', [('Shaka Khan', 'PERSON')]],
      ['I like London and Berlin.', [('London', 'LOCATION'), ('Berlin', 'LOCATION')]],
      ['There is a banyan tree in the courtyard', [('banyan tree', 'TREE')]],
      ['Dogs are more adorable than cats', [('Dogs', 'ANIMAL'), ('cats', 'ANIMAL')]], 
      ['John Watson is looking for his cap', [('John Watson', 'PERSON')]], 
      ['Beijing is the capital city of China', [('Beijing', 'LOCATION')]], 
      ['Leaves of Pine never yellow', [('Pine', 'TREE')]] 
    ], 
    columns=['text', 'annotation'])  
  createData(data, filepath_test)
  print()

pdb.set_trace()
print()