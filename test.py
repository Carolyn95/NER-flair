import pdb 
import pandas as pd 

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