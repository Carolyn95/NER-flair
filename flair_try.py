from flair.data import Corpus
from flair.datasets import ColumnCorpus
# Define columns
columns = {0: 'text', 1: 'pos', 2: 'ner'}
# Parent folder of train/test/dev files
data_folder = './dummy-data'
# Initialize a corpus using column format, data folder and names of train/dev/test file
corpus: Corpus = ColumnCorpus(data_folder,
                              columns,
                              train_file='train.txt',
                              test_file='test.txt',
                              dev_file='dev.txt')

print(len(corpus.train))
print(corpus.train[0].to_tagged_string('ner'))
# Tag to predict
tag_type = 'ner'
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
# Use flair embeddings
from flair.embeddings import WordEmbeddings, StackedEmbeddings, TokenEmbeddings
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
trainer.train('resources/taggers/example-ner',
              learning_rate=0.1,
              mini_batch_size=32,
              max_epochs=2)

# Use the trained model to predict
from flair.data import Sentence
from flair.models import SequenceTagger
model = SequenceTagger.load('resources/taggers/example-ner/final-model.pt')
sentence = Sentence('I love Berlin')
model.predict(sentence)
print(sentence.to_tagged_string())
