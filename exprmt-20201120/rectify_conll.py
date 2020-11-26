# 20200825
# FLAIR NER TRAINING
# corpus: Corpus = CONLL_03(base_path='reproduce_ner/tasks')
from __future__ import absolute_import, division, print_function
from flair.data import Corpus
from flair.datasets import CONLL_03
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, PooledFlairEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from typing import List
import argparse


def main(base_path, output_dir, nb_epochs):
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--data_dir", default='./', type=str, required=True, help="The parent dir of inpu data, must contain folder name `conll_03`")
    # parser.add_argument("--output_dir", default=None, required=True, help="The output directory where is going to store the trained model")
    # parser.add_argument("--train_epochs", default=3, type=int, required=True, help="Number of epochs to train")
    # args = parser.parse_args()
    # base_path = args.data_dir
    corpus: Corpus = CONLL_03(base_path=base_path)
    tag_type = 'ner'
    tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
    embedding_types: List[TokenEmbeddings] = [
        WordEmbeddings('glove'),
        PooledFlairEmbeddings('news-forward', pooling='min'),
        PooledFlairEmbeddings('news-backward', pooling='min'),
    ]

    embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)
    tagger: SequenceTagger = SequenceTagger(hidden_size=256,
                                            embeddings=embeddings,
                                            tag_dictionary=tag_dictionary,
                                            tag_type=tag_type)

    trainer: ModelTrainer = ModelTrainer(tagger, corpus)

    # output_dir = args.output_dir
    # nb_epochs = args.train_epochs
    # output_dir = 
    # nb_epochs = 10
    trainer.train(output_dir,
                train_with_dev=False,  
                max_epochs=nb_epochs) # 150 

if __name__ == '__main__':
    parent_dir = 'conll_frac/'
    # data_dir = ['10ptdata', '1ptdata', '3ptdata', '5ptdata', 'pt1ptdata', 'pt3ptdata', 'pt5ptdata']
    data_dir = ['100ptdata']
    data_dir = [parent_dir + d for d in data_dir]
    # models-20201124: 10e
    output_dir = [directory + '/models-5e-20201124' for directory in data_dir]
    nb_epochs = 10
    for dd, od in zip(data_dir, output_dir):
        print('='*10, dd, od, '='*10)
        main(dd, od, nb_epochs)
