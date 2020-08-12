# Code snippets, model trained on Conll-03, 4-class NER
from flair.models import SequenceTagger
from flair.data import Sentence

# Load the model
tagger = SequenceTagger.load('ner')
sentence = Sentence('George Washington went to Washington.')
# Predict NER tags
import pdb
pdb.set_trace()
tagger.predict(sentence)
print(sentence.to_tagged_string())
for entity in sentence.get_spans('ner'):
  print(entity)
print(sentence.to_dict(tag_type='ner'))
