# Neural Networks based Text Classification

Datasets and scripts for my Master's Thesis on:
Neural Network Based News Classification. An English-Polish Comparative Studies.

## Datasets

### CNN
CNN news are taken from [DeepMind Q&A Dataset](https://cs.nyu.edu/~kcho/DMQA/).

40,500 articles in 15 categories, 2500 training, 100 validation and 100 test files each. Categories are: crime, health, politics, showbiz, sport, tech, travel, us, africa, americas, asia, europe, middle east, living, opinion. Categories are build on keywords in urls.

### BBC
BBC news are taken from [BBC Datasets](http://mlg.ucd.ie/datasets/bbc.html).

2225 articles in 5 categories, 300 training, 50 validation and 50 test files each. Categories are: business, entertainment, politics, sport, tech. Source files are originally divided into categories.

### Rzeczpospolita
Rzeczpospolita news are taken from [Korpus "Rzeczpospolitej"](http://www.cs.put.poznan.pl/dweiss/research/rzeczpospolita/).

40,000 articles in 8 categories, 5000 training, 1000 validation and 1000 test files each. Categories are: Ekonomia, Prawo, Świat, Kraj, Sport, Gazeta, Kultura, Nauka i Technika. Categories are build on metadata included in source HTML files.

Rzeczpospolita source files are in HTML format so some analysis and cleaning is necessary. Only content text is taken, titles are ignored. Also text with tabular data are ignored.

## Word Embeddings

Custom word2vec models are build on cleaned source data to contain all words from train, validation and test files.

Pretrained models are used for comparison with custom ones:
- [Glove](https://nlp.stanford.edu/projects/glove/) for english texts (BBC and CNN) - Wikipedia 2014 + Gigaword 5 (6B tokens, 400K vocab, uncased, 100d vectors,
- [CLARIN-PL, Vector representations of polish words (Word2Vec method) ](https://clarin-pl.eu/dspace/handle/11321/327?show=full) for polish texts (Rzeczpospolita).

## Scripts

`settings.py` 

Folders and scripts configuration. 

`prepare.py --dataset <name> --w2v <model_name>`

Data preparation for given dataset. Dataset name choices are: _cnn_, _bbc_ and _rz_. 

Additionally when model name is provided Word2Vec model will be trained on dataset.

From CNN dataset only stories folder is used. Dataset contains also three files with article URLs. All this files should be concatenated into one file before processing.

`train.py --dataset <name> --nn-type <type> --model <model_name> --w2v <model_name> --batch-size <int> --epochs <int> --length <int>`

Training. Dataset name as above, neural network types are: _simple_ for standard feedforward network, _cnn_ for  convolution network and _rnn_ for LSTM network.

Model name is a file name to save model file. W2V model name is for word2vec model to be used for training.

Additional and optional parameters are: _batch-size_, _epochs_ and _length_. The first and the second are self explanatory, _length_ is for setting the length of articles to be used for training. Shorter ones are filled to the length with zero vectors, longer ones are cut to the defined length.

`results.py --dataset <name> --model <model_name> --batch-size <int> --length <int>`

Checking results on test dataset. Dataset and model name like in training.

`polish_w2v_cleaning.py`

Script for removing redundant info from polish word2vec model described in Word Embeddings section.