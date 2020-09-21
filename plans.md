# base environment: source ~/environments/torch_envr/bin/activate

* data preparation
fault report
statements that user may use to report faults
application names

* experiment
prepare folder 
prepare data 

## 20200812
* reproducibility
* 1 
  * reproduce turorial result
  * 4 entities: (ANIMAL, LOCATION, PERSON, TREE)
  * 1 sentence per entity
  * train = dev = test 
  * model settings: hidden_size=256, embedding: WordEmbeddings('glove')
  * hyperparams: learning_rate=.1, mini_batch_size=32, max_epochs=150
* 2  
  * reproduce turorial result
  * 4 entities: (ANIMAL, LOCATION, PERSON, TREE)
  * 2 sentences per entity
  * train = dev = test 
  * model settings: hidden_size=256, embedding: WordEmbeddings('glove')
  * hyperparams: learning_rate=.1, mini_batch_size=32, max_epochs=150
* 3 
  * modify turorial result
  * 4 entities: (APPLICATION, LOCATION, PERSON, DEVICE)
  * 1 sentences per entity
  * train = dev = test 
  * model settings: hidden_size=256, embedding: WordEmbeddings('glove')
  * hyperparams: learning_rate=.1, mini_batch_size=32, max_epochs=150
* 4
  * modify turorial result
  * 4 entities: (APPLICATION, LOCATION, PERSON, DEVICE)
  * 2 sentences per entity
  * train = dev = test 
  * model settings: hidden_size=256, embedding: WordEmbeddings('glove')
  * hyperparams: learning_rate=.1, mini_batch_size=32, max_epochs=150
* 5 
  * modify turorial result
  * 4 entities: (APPLICATION, LOCATION, PERSON, DEVICE)
  * 2 sentences per entity
  * train, dev, test 
  * model settings: hidden_size=256, embedding: WordEmbeddings('glove')
  * hyperparams: learning_rate=.1, mini_batch_size=32, max_epochs=150
* 6
  * reproduce turorial result
  * 4 entities: (ANIMAL, LOCATION, PERSON, TREE)
  * 1 sentence per entity
  * train, dev, test 
  * model settings: hidden_size=256, embedding: WordEmbeddings('glove')
  * hyperparams: learning_rate=.1, mini_batch_size=32, max_epochs=150


## 20200917
* add argparse to the script of generating synthetic data
`python create_data.py --tmpl_name=text_devices.tmpl --tag_type=DEVICE --num_sample=150`

* sample synthetic data to create training dataset

* 3 exprmts 
  * exprmt 1: 150 (smp/ent) x 4 (ent) = 600 smp, take fractions (2.5% ~ 15, 5%, 10%, 20%) from whole dataset as training set, rest for testing
  * exprmt 2: create entirely unseen dataset (not using tmpl) for testing -> breaking test, don't need to re-train above model, just use it as it is to test on this unseen testing set
  * exprmt 3: use all sampling data to train model, test on exprmt 2's testing set
