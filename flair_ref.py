# ref: https://medium.com/thecyphy/training-custom-ner-model-using-flair-df1f9ea9c762
# Sentence tokenizer and create data
import random 
import pandas as pd
from tqdm import tqdm
from difflib import SequenceMatcher
import re
import pickle as plk
import numpy as np 
import torch


def setSeed(lucky_number): 
  np.random.seed(lucky_number)
  random.seed(lucky_number)
  torch.manual_seed(lucky_number)
  torch.cuda.manual_seed(lucky_number)
  torch.cuda.manual_seed_all(lucky_number)
  torch.backends.cudnn.enabled = False
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True

setSeed(2020)
# BATCH_SIZE = 32
# torch.utils.data.DataLoader(training, shuffle = True, batch_size=BATCH_SIZE, worker_init_fn=np.random.seed(0),num_workers=0)

def matchSequence(string, pattern):
  """Return start and end index of any pattern present in the text
  """
  match_list = []
  pattern = pattern.strip()
  seq_match = SequenceMatcher(None, string, pattern, autojunk=False)
  match = seq_match.find_longest_match(0, len(string), 0, len(pattern))
  if (match.size == len(pattern)):
    start = match.a
    end = match.a + match.size
    match_tup = (start, end)
    string = string.replace(pattern, "X" * len(pattern), 1)
    match_list.append(match_tup)

  return match_list, string


def markSentence(s, match_list):
  """Marks all the entities in the sentence as per the BIO scheme
  """
  word_dict = {}
  for word in s.split():
    word_dict[word] = '0'

  for start, end, e_type in match_list:
    temp_str = s[start:end]
    tmp_list = temp_str.split()
    if len(tmp_list) > 1:
      word_dict[tmp_list[0]] = 'B-' + e_type
      for w in tmp_list[1:]:
        word_dict[w] = 'I-' + e_type
    else:
      word_dict[temp_str] = 'B-' + e_type
  return word_dict


def clean(text):
  """Helper function to add a space before the puntuations for better tokenization
  """
  filters = [
      '!', '#', '$', '%', '&', '(', ')', '/', '*', '.', ':', ';', '<', '=', '>',
      '?', '@', '[', '\\', ']', '_', '`', '{', '}', '~', "'"
  ]
  for i in text:
    if i in filters:
      text = text.replace(i, ' ' + i)
  return text


def createData(df, filepath):
  """Create data in the desired format
  """
  with open(filepath, 'w') as f:
    for text, annotation in zip(df.text, df.annotation):
      text = clean(text)
      text_ = text
      match_list = []
      for i in annotation:
        a, text_ = matchSequence(text, i[0])
        match_list.append((a[0][0], a[0][1], i[1]))

      d = markSentence(text, match_list)
      for i in d.keys():
        f.writelines(i + ' ' + d[i] + '\n')
      f.writelines('\n')


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

def trainModel(serial_no):
  # define columns
  columns = {0 : 'text', 1 : 'ner'}
  # directory where the data resides
  data_folder = 'dummy-data/dummy-data-' + str(serial_no) + '/'
  # initializing the corpus
  from flair.datasets import ColumnCorpus
  corpus: Corpus = ColumnCorpus(data_folder, columns,
                                train_file = 'train.txt',
                                test_file = 'test.txt',
                                dev_file = 'dev.txt')


  # Tag to predict
  tag_type = 'ner'
  tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
  # Use flair embeddings
  from flair.embeddings import WordEmbeddings, StackedEmbeddings
  from typing import List
  embedding_types: List[TokenEmbeddings] = [
      WordEmbeddings('glove'),
  ]
  embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)
  # Initialize sequence tagger (bi-LSTM, CRF)
  from flair.models import SequenceTagger
  tagger: SequenceTagger = SequenceTagger(hidden_size=256,
                                          embeddings=embeddings,
                                          tag_dictionary=tag_dictionary,
                                          tag_type=tag_type,
                                          use_crf=True)
  print(tagger)

  # Train model
  from flair.trainers import ModelTrainer
  trainer: ModelTrainer = ModelTrainer(tagger, corpus)
  trainer.train('dummy-model/dummy-model-' + str(serial_no),
                learning_rate=0.1,
                mini_batch_size=32,
                max_epochs=150)

def testModel(serial_no, test_sent):
  # Use the trained model to predict
  from flair.data import Sentence
  from flair.models import SequenceTagger
  modelpath = 'dummy-model/dummy-model-' + str(serial_no) + '/final-model.pt'
  model = SequenceTagger.load(modelpath)
  sentence = Sentence(test_sent)
  model.predict(sentence)
  print(sentence.to_tagged_string())

if __name__ == '__main__':
  # make sure the folder exists prior to execute this line
  # makeData()
  # trainModel(2)
  testModel(2, 'Pine is taller than banyan ')
