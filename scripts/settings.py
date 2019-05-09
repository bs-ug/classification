DATA_DIR = "../data/"
# CNN
# CNN_TOPICS = {
#     0: ["/crime/"],
#     1: ["/health/"],
#     2: ["/politics/"],
#     3: ["/showbiz/"],
#     4: ["/sport/"],
#     5: ["/tech/"],
#     6: ["/travel/"],
#     7: ["/us/"],
#     8: ["/africa/", "/world/africa/"],
#     9: ["/world/americas/"],
#     10: ["/world/asiapcf/", "/asia/", "/world/asia/"],
#     11: ["/europe/", "/world/europe/"],
#     12: ["/middleeast/", "/world/meast/"],
#     13: ["/living/"],
#     14: ["/opinion/", "/opinions/"]
# }
CNN_TOPICS = {
    0: ["/us/", "/world/americas/", "/US/law/", "/justice/"],
    1: ["/africa/", "/asia/", "/europe/", "/middleeast/",
        "/world/africa/", "/world/asiapcf/", "/world/asia/", "/world/europe/", "/world/meast/"]
}
CNN_DATA_DIR = "../data/cnn"
CNN_SOURCE_URLS_FILE = "wayback_urls.txt"
CNN_SOURCE_FILES = "../data/cnn/stories"
CNN_TRAINING_LABELS = "train.json"
CNN_VALIDATION_LABELS = "validation.json"
CNN_TEST_LABELS = "test.json"
CNN_TRAIN_QUANTITY = 10000
CNN_VALIDATION_QUANTITY = 1000
CNN_TEST_QUANTITY = 1000
CNN_TRAINING_FILES_PATH = "../data/cnn/train"
CNN_VALIDATION_FILES_PATH = "../data/cnn/validation"
CNN_TEST_FILES_PATH = "../data/cnn/test"
CNN_MODEL_PATH = "../data/cnn/model"
CNN_MODEL_NAME = "cnn_w2v_300.model"
CNN_PICKLE_PATH = "../data/cnn/pickle"
CNN_MIN_ARTICLE_LENGTH = 200

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
POLISH_TRAIN_QUANTITY = 10000
POLISH_VALIDATION_QUANTITY = 1000
POLISH_TEST_QUANTITY = 1000
POLISH_TRAINING_FILES_PATH = "../data/polish/train"
POLISH_VALIDATION_FILES_PATH = "../data/polish/validation"
POLISH_TEST_FILES_PATH = "../data/polish/test"
POLISH_MODEL_PATH = "../data/polish/model"
POLISH_MODEL_NAME = "polish_w2v_300.model"
POLISH_MIN_ARTICLE_LENGTH = 200

RZ_TOPICS = {
    "gazeta/Ekonomia": 0,
    "gazeta/Prawo": 1,
    "gazeta/Świat": 2,
    "gazeta/Kraj": 3,
    "gazeta/Sport": 4,
    "gazeta/Gazeta": 5,
    "gazeta/Kultura": 6,
    "gazeta/Nauka i Technika": 7
}
RZ_DATA_DIR = "../data/rz"
RZ_LABELS = "labels.json"
RZ_SOURCE_FILES = "../data/rz/source"
RZ_MODEL_PATH = "../data/rz/model"
RZ_MODEL_NAME = "rz_w2v_300.model"
RZ_TRAINING_LABELS = "train.json"
RZ_VALIDATION_LABELS = "validation.json"
RZ_TEST_LABELS = "test.json"
RZ_TRAINING_FILES_PATH = "../data/rz/train"
RZ_VALIDATION_FILES_PATH = "../data/rz/validation"
RZ_TEST_FILES_PATH = "../data/rz/test"
RZ_TRAIN_QUANTITY = 5000
RZ_VALIDATION_QUANTITY = 1000
RZ_TEST_QUANTITY = 1000
RZ_MIN_ARTICLE_LENGTH = 200


EMBEDDINGS_VECTOR_LENGTH = 300
