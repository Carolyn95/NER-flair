from flair.models import SequenceTagger
tagger = SequenceTagger.load("/home/carolyn/.flair/models/en-ner-conll03-v0.4.pt")
from flair.data import Corpus
from flair.datasets import ColumnCorpus
columns = {0: 'text', 1: 'ner'}                                                                                                                                    
data_folder = './dummy-data/'                                                                                       
corpus: Corpus = ColumnCorpus(data_folder, columns)                                                                     
tag_type = 'ner'                                                                                                                                                   
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)   
from flair.trainers import ModelTrainer                                                                                                                            
trainer: ModelTrainer = ModelTrainer(tagger, corpus)                     
trainer.train('dummy-model', train_with_dev=True, max_epochs=5)              
model = SequenceTagger.load('dummy-model/final-model.pt')  
