DATA_DIR = "data"
TRAINING_LABELS = "train.json"
VALIDATION_LABELS = "validation.json"
TEST_LABELS = "test.json"
TRAINING_FILES = "train"
VALIDATION_FILES = "validation"
TEST_FILES = "test"
MODELS_PATH = "data/models"
EMBEDDINGS_VECTOR_LENGTH = 100
PADDING_LENGTH = 400

# CNN
CNN = "cnn"
CNN_TOPICS = {
    0: ['/world/'],
    1: ['/us/'],
    2: ['/politics/'],
    3: ['/sport/'],
    4: ['/opinion/', '/opinions/'],
    5: ['/showbiz/', '/entertainment/']
}
CNN_DATA_DIR = "data/cnn"
CNN_SOURCE_URLS_FILE = "wayback_urls.txt"
CNN_SOURCE_FILES = "data/cnn/stories"
CNN_TRAIN_QUANTITY = 5000 # 10000
CNN_VALIDATION_QUANTITY = 1000 # 1000
CNN_TEST_QUANTITY = 1000 # 1000
CNN_MIN_ARTICLE_LENGTH = 100

# BBC
BBC = "bbc"
BBC_TOPICS = {
    0: "business",
    1: "entertainment",
    2: "politics",
    3: "sport",
    4: "tech"
}
BBC_DATA_DIR = "data/bbc"
BBC_TRAIN_QUANTITY = 300
BBC_VALIDATION_QUANTITY = 40
BBC_TEST_QUANTITY = 40
BBC_MIN_ARTICLE_LENGTH = 100

# Rzeczpospolita news
RZ = "rz"
RZ_TOPICS = {
    "gazeta/Świat": 0,
    "gazeta/Kraj": 1,
    "gazeta/Prawo": 2,
    "gazeta/Sport": 3,
    "gazeta/Kultura": 4,
    "gazeta/Nauka i Technika": 5
}
RZ_DATA_DIR = "data/rz"
RZ_LABELS = "labels.json"
RZ_SOURCE_FILES = "data/rz/source"
RZ_TRAIN_QUANTITY = 5000
RZ_VALIDATION_QUANTITY = 1000
RZ_TEST_QUANTITY = 1000
RZ_MIN_ARTICLE_LENGTH = 100
