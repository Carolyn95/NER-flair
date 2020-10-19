# 20200917
import os 
import numpy as np 
import pdb
from pathlib import Path
import random  
from collections import Counter

seed = 2020
random.seed(seed)
np.random.seed(seed)

def readDataFolder(data_folder, is_randomized=True, sample_rate=1, save_path=None): 
  """Read separate data from a data folder
  # data_folder, is_randomized, sample_rate, save_path
  sample_rate, stractified sampling
  save a file
  TODO: take care of/optimize sample_rate >= 1
  """
  if save_path:
    save_path = Path(save_path)
  if not os.path.exists(save_path):
    os.mkdir(save_path)

  data_folder = Path(data_folder)
  all_data, all_traindata, all_testdata = [], [], []
  for filename in os.listdir(data_folder):
    with open(data_folder / filename, 'r') as f:
      data = list(filter(None, f.read().split('\n\n')))
    if is_randomized:
      data_train = random.sample(data, int(len(data) * sample_rate))
      if sample_rate < 1:
        data_test = [d for d in data if d not in data_train]
    else:
      data_train = data[:int(len(data) * sample_rate)]
      if sample_rate < 1:
        data_test = data[int(len(data)) * sample_rate:]
    all_data += data
    np.save(save_path / 'strf_sampled_alldata.npy', all_data)
    if sample_rate < 1:
      all_traindata += data_train 
      all_testdata += data_test 
      np.save(save_path / 'strf_sampled_alldata_train.npy', all_traindata)
      np.save(save_path / 'strf_sampled_alldata_test.npy', all_testdata)


def readSingle(data_path, sample_rate=1, save_path=None): 
  """
  not stractified sampling, basically random sampling
  """
  if save_path:
    save_path = Path(save_path)
  if not os.path.exists(save_path):
    os.mkdir(save_path)

  with open(data_path, 'r') as f:
    data = list(filter(None, f.read().split('\n\n')))
  data_train = random.sample(data, int(len(data) * sample_rate))
  np.save(save_path / (os.path.split(data_path)[-1].split('.')[0] + '_train'), data_train)
  if sample_rate < 1:
    data_test = [d for d in data if d not in data_train]
    np.save(save_path / (os.path.split(data_path)[-1].split('.')[0] + '_test'), data_test)

"""
def strictSample(data_folder, is_randomized=True, sample_rate=1, save_path=None):
  # aim at fixing test set and sample sub-samples from sub-training set 
  # data_folder: temp data folder, containing sub-training and testing
  # sub-training is exactly 20pct of synthetic data
  # sample rate: [1, 1/2, 1/4, 1/8]
  # naming: ['20pct', '10pct', '5pct', '2pt5pct']
  # mapping = {1: '20pct', 1/2: '10pct', 1/4: '5pct', 1/8: '2pt5pct'}
  data_folder = Path(data_folder)
  if save_path:
    save_path = Path(save_path)
  if not os.path.exists(save_path):
    os.mkdir(save_path)

  for filename in os.listdir(data_folder):
    if 'test' in filename:
      test_data = np.load(data_folder / filename, allow_pickle=True)
      np.save(save_path / 'test.npy', test_data)  
    elif 'train' in filename:
      train_data = np.load(data_folder / filename, allow_pickle=True)
      if is_randomized:
        # pdb.set_trace()
        sub_train = random.sample(train_data.tolist(), int(len(train_data) * sample_rate))
        np.save(save_path / 'train.npy', sub_train)
      else:
        sub_train = train_data[:int(len(train_data) * sample_rate)]
        np.save(save_path / 'train.npy', sub_train)
    else: 
      print('Not target file')
  
  print()
"""

def strictSample(data_folder, sample_rate, save_path):
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
        if temp and len(temp) == train_cats_counter[c]:
          sample_temp = random.sample(temp, int(len(temp) * sample_rate))
          samples[c] = sample_temp
      sampled_train = []
      for key, val in samples.items():
        temp = train_data[val].tolist()
        sampled_train += temp
      np.save(save_path / 'train.npy', sampled_train)
    else:
      print('Not target file')


def prepareSmallPercentage(test_ds, train_ds, sample_rate, save_path):
  """
  * save test_ds in save_path
  * sample train_ds by category and sample_rate
  * save resampled train_df in save_path
  """
  if not os.path.exists(save_path):
    os.mkdir(save_path)
  save_path = Path(save_path)
  np.save(save_path / 'test.npy', test_ds)

  def categorize(x):
    if 'APPLICATION' in x:
      return 'APPLICATION'
    elif 'DEVICE' in x:
      return 'DEVICE'
    elif 'LOCATION' in x:
      return 'LOCATION'
    else:
      raise EXCEPTION("Not recognisable entity")

  train_cats = [categorize(x) for x in train_ds]
  train_cats_counter = Counter(train_cats)
  cats = []
  samples = {}
  temp = []
  for i, c in enumerate(sorted(train_cats)):
    if c not in cats:
      cats.append(c)
      temp = []
    temp.append(i)      
    if temp and len(temp) == train_cats_counter[c]:
      sample_temp = random.sample(temp, int(len(temp) * sample_rate))
      samples[c] = sample_temp
  sampled_train = []
  for key, val in samples.items():
    temp = train_ds[val].tolist()
    sampled_train += temp
  np.save(save_path / 'train.npy', sampled_train)


if __name__ == '__main__':
  # sample rate list: [0.025, 0.05, 0.1, 0.2] -> [2.5%, 5%, 10%, 20%]
  # corresponding dir: [2pt5pct, 5pct, 10pct, 20pct]
  # readDataFolder('synthetic-data', True, 0.2, 'tmp')
  # readSingle('synthetic-data/text_apps.txt', 1, 'test_toyscript2')
  # mapping = {1: '20pct', 1/2: '10pct', 1/4: '5pct', 1/8: '2pt5pct'}
  # strictSample('tmp', 1, '20pct')
  train_ds = np.load('20pct/train.npy', allow_pickle=True) 
  test_ds = np.load('20pct/test.npy', allow_pickle=True)
  # sample_rate and save_path mapping: {1: '20pct', 1/2: '10pct', 1/4: '5pct', 1/8: '2pt5pct}
  prepareSmallPercentage(test_ds, train_ds, 1/8, '2pt5pct')
  print()
