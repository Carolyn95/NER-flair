import torch 
device = ('cuda' if torch.cuda.is_available() else 'cpu')
print(f'{device}')

# set seed in 3 lines to get reproducible results
import random
import torch 
import numpy as np 
random.seed(2020)
np.random.seed(2020)
torch.manual_seed(2020)
