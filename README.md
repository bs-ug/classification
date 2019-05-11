# Neural Networks based Text Classification

A comparison of neural network based text classification of news articles in English and Polish languages.

## Data

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

