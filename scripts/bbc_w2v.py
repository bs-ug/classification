import os

from gensim.models import Word2Vec

from scripts import settings
from scripts.utils import text_generator, W2VCallback


iterable = text_generator(os.path.join(settings.BBC_DATA_DIR, '*', "*.txt"))
model = Word2Vec([item for item in iterable],
                 size=settings.EMBEDDINGS_VECTOR_LENGTH,
                 window=5, min_count=1, workers=4,
                 compute_loss=True,
                 callbacks=[W2VCallback()])
model.wv.save_word2vec_format(os.path.join(settings.MODELS_PATH,
                                           f"{settings.BBC}_{settings.EMBEDDINGS_VECTOR_LENGTH}.w2v"),
                              binary=False)