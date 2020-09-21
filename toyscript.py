# 20200917
import os 
import numpy as np 
import pdb
from pathlib import Path
import random  


random.seed(2020)
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


if __name__ == '__main__':
  # sample rate list: [0.025, 0.05, 0.1, 0.2] -> [2.5%, 5%, 10%, 20%]
  # corresponding dir: [2pt5pct, 5pct, 10pct, 20pct]
  readDataFolder('synthetic-data', True, 0.2, '20pct')
  # readSingle('synthetic-data/text_apps.txt', 1, 'test_toyscript2')
  print()
