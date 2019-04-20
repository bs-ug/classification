import os

from scripts import settings
from scripts.cnn_datasets import prepare_dataset
from scripts.cnn_embedings import get_word_embeddings

train_x, train_y = prepare_dataset(
    os.path.join(settings.CNN_DATA_DIR, settings.CNN_TRAINING_LABELS),
    settings.CNN_TRAINING_FILES_PATH)
validation_x, validation_y = prepare_dataset(
    os.path.join(settings.CNN_DATA_DIR, settings.CNN_VALIDATION_LABELS),
    settings.CNN_VALIDATION_FILES_PATH)
test_x, test_y = prepare_dataset(
    os.path.join(settings.CNN_DATA_DIR, settings.CNN_TEST_LABELS),
    settings.CNN_TEST_FILES_PATH)

embeddings = get_word_embeddings(os.path.join(settings.CNN_MODEL_PATH, settings.CNN_MODEL_NAME), train_x, validation_x)
