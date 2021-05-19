# 20200917
import os
import numpy as np
import pdb
from pathlib import Path
import random
from collections import Counter

seed = 2021
random.seed(seed)
np.random.seed(seed)


def subsetData(data_dir, subset_no, save_dir):
  data_dir, save_dir = Path(data_dir), Path(save_dir)
  if not os.path.exists(save_dir):
    os.mkdir(save_dir)
  for filename in os.listdir(data_dir):
    with open(data_dir / filename, 'r') as f:
      data = list(filter(None, f.read().split('\n\n')))
    if 'train' in filename:
      data = data[:subset_no]
    with open(save_dir / filename, 'w') as f:
      for d in data:
        f.write(d)
        f.write('\n\n')
  print('Subsetting finished.')


if __name__ == '__main__':
  data_dir = 'conll-full'
  subset_no = 1500
  save_dir = 'conll_1500'
  subsetData(data_dir, subset_no, save_dir)
