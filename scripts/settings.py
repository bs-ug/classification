# CNN
CNN_TOPICS = {
    0: ["/crime/"],
    1: ["/health/"],
    2: ["/politics/"],
    3: ["/showbiz/"],
    4: ["/sport/"],
    5: ["/tech/"],
    6: ["/travel/"],
    7: ["/us/"],
    8: ["/africa/", "/world/africa/"],
    9: ["/world/americas/"],
    10: ["/world/asiapcf/", "/asia/", "/world/asia/"],
    11: ["/europe/", "/world/europe/"],
    12: ["/middleeast/", "/world/meast/"],
    13: ["/living/"],
    14: ["/opinion/", "/opinions/"]
}
CNN_DATA_DIR = "../data/cnn"
CNN_URLS_FILE = "wayback_training_urls.txt"
CNN_SOURCE_FILES = "../data/cnn/stories"
CNN_TRAINING_LABELS = "train.json"
CNN_VALIDATION_LABELS = "validation.json"
CNN_TEST_LABELS = "test.json"
CNN_TRAIN_QUANTITY = 2500
CNN_VALIDATION_QUANTITY = 100
CNN_TEST_QUANTITY = 100
CNN_TRAINING_FILES_PATH = "../data/cnn/train"
CNN_VALIDATION_FILES_PATH = "../data/cnn/validation"
CNN_TEST_FILES_PATH = "../data/cnn/test"
CNN_MODEL_PATH = "../data/cnn/model"
CNN_MODEL_NAME = "cnn_w2v_300.model"
CNN_PICKLE_PATH = "../data/cnn/pickle"

# Polish news
POLISH_TOPICS = {
    0: ["wiadomosci.onet.pl/swiat/",
        "fakty.interia.pl/swiat/",
        "www.tvn24.pl/wiadomosci-ze-swiata",
        "www.fakt.pl/wydarzenia/swiat/",
        "www.wprost.pl/swiat/",
        "www.rmf24.pl/fakty/swiat/"],
    1: ["wiadomosci.onet.pl/kraj/",
        "fakty.interia.pl/polska/",
        "www.tvn24.pl/wiadomosci-z-kraju",
        "www.fakt.pl/wydarzenia/polska/",
        "www.wprost.pl/kraj/",
        "www.rmf24.pl/fakty/polska/"]
}
POLISH_DATA_DIR = "../data/polish"
POLISH_SOURCE_URLS_FILE = "links.txt"
POLISH_SOURCE_FILES = "../data/polish/source"
POLISH_FILTERED_FILES = "../data/polish/filtered"
POLISH_TRAINING_LABELS = "train.json"
POLISH_VALIDATION_LABELS = "validation.json"
POLISH_TEST_LABELS = "test.json"
POLISH_TRAIN_QUANTITY = 19000
POLISH_VALIDATION_QUANTITY = 900
POLISH_TEST_QUANTITY = 900
POLISH_TRAINING_FILES_PATH = "../data/polish/train"
POLISH_VALIDATION_FILES_PATH = "../data/polish/validation"
POLISH_TEST_FILES_PATH = "../data/polish/test"
POLISH_MODEL_PATH = "../data/polish/model"
POLISH_MODEL_NAME = "polish_w2v_300.model"

EMBEDDINGS_VECTOR_LENGTH = 300
