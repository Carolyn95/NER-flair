# 20200921
# try train model on synthetic data
# sanity check first
## read data and save them as conll_03 format in order to reuse *reproduce_ner.py*

import numpy as np 
import os 
from pathlib import Path 
import random
import torch

seed = 2020
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

from flair.data import Corpus, Sentence
# from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, PooledFlairEmbeddings
from flair.embeddings import TransformerWordEmbeddings, StackedEmbeddings, WordEmbeddings
from typing import List
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from flair.datasets import ColumnCorpus
import argparse
import pdb 

def convertData(data_dir, save_dir):
  # data_dir, save_dir
  # read from numpy array format and convert to txt format and name as 'eng.train' if train data, 'eng.testa' if dev data, 'eng.testb' if test data
  save_dir = Path(save_dir)
  if not os.path.exists(save_dir):
    os.mkdir(save_dir)

  data_dir = Path(data_dir)
  # DOC_START_STRING = {'train': '', 'dev': '', 'test': ''}
  for data_filename in os.listdir(data_dir):
    if 'train' in data_filename:
      data = np.load(data_dir / data_filename, allow_pickle=True)
      with open(save_dir / 'train.txt', 'w') as f:
        # f.write('-DOCSTART- -X- O O\n\n')
        for d in data:
          f.write(d)
          f.write('\n\n')

    if 'test' in data_filename:
      data = np.load(data_dir / data_filename, allow_pickle=True)
      with open(save_dir / 'test.txt', 'w') as f:
        # f.write('-DOCSTART- -X- -X- O\n\n')
        for d in data:
          f.write(d)
          f.write('\n\n')

    if 'dev' in data_filename:
      data = np.load(data_dir / data_filename, allow_pickle=True)
      with open(save_dir / 'dev.txt', 'w') as f:
        # f.write('-DOCSTART- -X- O O\n\n')
        for d in data:
          f.write(d)
          f.write('\n\n')        

  print('Convertion done!')


def trainNER(data_dir, model_dir):
  parser = argparse.ArgumentParser()
  parser.add_argument("--model", default='bert-base-cased', type=str, required=True, help="The pretrained model to produce embeddings")
  args = parser.parse_args()
  model = args.model
  columns = {0 : 'text', 1 : 'ner'}
  # pdb.set_trace()
  # print(data_dir + '/eng.train')
  corpus: Corpus = ColumnCorpus(data_dir, columns)
  corpus.filter_empty_sentences()
  tag_type = 'ner'
  # tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
  tag_dictionary = corpus.make_label_dictionary('ner')
  print(tag_dictionary.get_items())
  stats = corpus.obtain_statistics()
  print(stats)
  # ['<unk>', 'O', 'B-DEVICE', 'I-DEVICE', 'B-TREE', 'I-TREE', 'B-APPLICATION', 'I-APPLICATION', 'B-LOCATION', 'I-LOCATION', '<START>', '<STOP>']
  # pdb.set_trace()

  embedding_types: List[TokenEmbeddings] = [
    WordEmbeddings('glove'), 
    TransformerWordEmbeddings(
          model=model, 
          layers='0',  # dtype: str
          pooling_operation='first_last', 
          use_scalar_mix=False,
          batch_size=16,
          fine_tune=False,
          allow_long_sentences=False
          )]
  embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)


  tagger: SequenceTagger = SequenceTagger(hidden_size=256,
                                          embeddings=embeddings,
                                          tag_dictionary=tag_dictionary,
                                          tag_type=tag_type)

  trainer: ModelTrainer = ModelTrainer(tagger, corpus)

  trainer.train(model_dir,
                train_with_dev=True,  
                max_epochs=10) # 150 


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
  model = SequenceTagger.load(model_path)
  
  if test_sent:
    print('Predicting in singular mode')
    test_sent = Sentence(test_sent)
    model.predict(test_sent)
    print(test_sent.to_tagged_string())

  if test_file_dir:
    print('Predicting in plural mode')
    try:
      columns = {0: 'text', 1: 'ner'}
      corpus = ColumnCorpus(test_file_dir, columns)
      test_data = corpus.test 
    except:
      raise Exception('Directory must contain `test.txt` file, one column `text` the other `ner`')
    
    test_result, test_loss = model.evaluate(test_data, out_path=test_file_dir + '/test.tsv')
    result_line = f"\t{test_loss}\t{test_result.log_line}"
    print(f"TEST : loss {test_loss} - score {round(test_result.main_score, 4)}")
    print(f"TEST RESULT : {result_line}")

  print('end')

if __name__ == '__main__':
  # convertData('20pct', '20pct/data')
  # trainNER('20pct/data', '20pct/models')
  # testModel('20pct/models', test_file_dir='20pct/data')
  testModel('10pct/models', test_file_dir='20pct/data')
