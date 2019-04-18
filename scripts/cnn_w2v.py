from gensim.models import Word2Vec, KeyedVectors
from glob import glob
import os
from keras.preprocessing import text, sequence

CNN_SOURCE_FILES_PATH = "../data/cnn/cos"


def text_generator():
    for file in glob(os.path.join(CNN_SOURCE_FILES_PATH, "*.txt")):
        with open(file, "r", encoding="utf-8") as input_file:
            content = input_file.read()
        yield text.text_to_word_sequence(content)


iterable = text_generator()
model = Word2Vec([item for item in iterable], size=100, window=5, min_count=1, workers=4)
model.wv.save_word2vec_format("cnn_w2v.model", binary=False)
