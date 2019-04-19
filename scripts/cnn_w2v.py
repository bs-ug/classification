from gensim.models import Word2Vec, KeyedVectors
from gensim.models.callbacks import CallbackAny2Vec
from glob import glob
import os
from keras.preprocessing import text, sequence

CNN_SOURCE_FILES_PATH = "../data/cnn/train"


class Callback(CallbackAny2Vec):
    def __init__(self):
        self.epoch = 0

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        print('Loss after epoch {}: {}'.format(self.epoch, loss))
        self.epoch += 1

    def on_epoch_start(self, model):
        print(f"{self.epoch} started.")

    def on_train_begin(self, model):
        print("Training started.")


def text_generator():
    for file in glob(os.path.join(CNN_SOURCE_FILES_PATH, "*.txt")):
        with open(file, "r", encoding="utf-8") as input_file:
            content = input_file.read()
        yield text.text_to_word_sequence(content)


iterable = text_generator()
model = Word2Vec([item for item in iterable], size=100, window=5, min_count=1, workers=4, compute_loss=True, callbacks=[Callback()])
model.wv.save_word2vec_format("cnn_w2v.model", binary=False)
