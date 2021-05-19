import numpy as np
import pdb
# All possible datasets (incl. conll03)
# need to configure data path in the code


def filterByEntityType(data_path, ent_type, new_ent_type):
  # read data from numpy array
  data = np.load(data_path, allow_pickle=True)
  filtered_data = []
  for d in data:
    if ent_type in d:
      new_d = d.replace(ent_type, new_ent_type)
      filtered_data.append(new_d)
  filtered_data = np.array(filtered_data)
  return filtered_data


def saveData(data, save_path):
  with open(save_path, 'w') as f:
    for d in data:
      f.write(d)
      f.write('\n\n')
  print('---')


def flairInfer(model_path, test_or_train):
  from flair.data import Sentence
  from flair.models import SequenceTagger
  from flair.data import Corpus
  from flair.datasets import CONLL_03
  from flair.datasets import ColumnCorpus

  model = SequenceTagger.load(model_path + '/final-model.pt')
  data_dir = '../GmbDataExperimentation/processed_data/1500_data'
  try:

    corpus: Corpus = CONLL_03(base_path=data_dir)

  except:
    pass
    columns = {0: 'text', 1: 'ner'}
    corpus: Corpus = ColumnCorpus(data_dir, columns)
  if test_or_train == 'train':
    testdata = corpus.train
    result_file = data_dir + '/train.tsv'
  else:
    testdata = corpus.test
    result_file = data_dir + '/test.tsv'

  test_result, test_loss = model.evaluate(testdata, out_path=result_file)
  result_line = f"\t{test_loss}\t{test_result.log_line}"
  print(f"TEST : loss {test_loss} - score {round(test_result.main_score, 4)}")
  print(f"TEST RESULT : {result_line}")


if __name__ == '__main__':
  # filtering 'LOC' from test and train
  # test_data_path = '../2pt5pct/test.npy'
  # ent_type = 'LOCATION'
  # new_ent_type = 'LOC'
  # test_data = filterByEntityType(test_data_path, ent_type, new_ent_type)
  # print('Test data length is {}'.format(len(test_data)))
  # saveData(test_data, '../2pt5pct/loc_data/test.txt')
  # # pdb.set_trace()
  # train_data_path = '../2pt5pct/train.npy'
  # train_data = filterByEntityType(train_data_path, ent_type, new_ent_type)
  # print('Train data length is {}'.format(len(train_data)))
  # saveData(train_data, '../2pt5pct/loc_data/train.txt')

  # model_path = '../GmpDataExperimentation/taggers'
  # model_path = '../GmpDataExperimentation/processed_data/15_data/models'
  model_path = './GmbData/1500_data/models'
  flairInfer(model_path, 'train')  # train | test
