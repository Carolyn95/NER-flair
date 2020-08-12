import os 
import pdb 
import numpy 
import random 


datadir = 'synthetic-data/'
files = os.listdir(datadir)

for file in files:
  with open(datadir + file, 'r') as f:
    data = f.read()
    data = [_ for _ in data.split('\n\n') if _]
    
    pdb.set_trace()
    print()
def splitData():
  print()