import pdb 
import pandas as pd 
from pathlib import Path 
import os 
import numpy as np 
from collections import Counter
import random

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