DATA_DIR = "../data/"
TRAINING_LABELS = "train.json"
VALIDATION_LABELS = "validation.json"
TEST_LABELS = "test.json"
TRAINING_FILES = "train"
VALIDATION_FILES = "validation"
TEST_FILES = "test"
MODELS_PATH = "../data/models"
EMBEDDINGS_VECTOR_LENGTH = 300

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
# CNN_TOPICS = {
#     0: ["/us/", "/world/americas/", "/US/law/", "/justice/"],
#     1: ["/africa/", "/asia/", "/europe/", "/middleeast/",
#         "/world/africa/", "/world/asiapcf/", "/world/asia/", "/world/europe/", "/world/meast/"]
# }
CNN_DATA_DIR = "../data/cnn"
CNN_SOURCE_URLS_FILE = "wayback_urls.txt"
CNN_SOURCE_FILES = "../data/cnn/stories"
CNN_TRAIN_QUANTITY = 2500 # 10000
CNN_VALIDATION_QUANTITY = 100 # 1000
CNN_TEST_QUANTITY = 100 # 1000
CNN_MODEL_NAME = "cnn_300.w2v"
CNN_MIN_ARTICLE_LENGTH = 200

# BBC
BBC_TOPICS = {
    0: "business",
    1: "entertainment",
    2: "politics",
    4: "sport",
    5: "tech"
}
BBC_DATA_DIR = "../data/bbc"
BBC_MODEL_NAME = "bbc_300.w2v"
BBC_TRAIN_QUANTITY = 300
BBC_VALIDATION_QUANTITY = 50
BBC_TEST_QUANTITY = 50
BBC_MIN_ARTICLE_LENGTH = 200

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
POLISH_TRAIN_QUANTITY = 10000
POLISH_VALIDATION_QUANTITY = 1000
POLISH_TEST_QUANTITY = 1000
POLISH_MODEL_NAME = "polish_300.w2v"
POLISH_MIN_ARTICLE_LENGTH = 200

# Rzeczpospolita news
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
RZ_MODEL_NAME = "rz_300.w2v"
RZ_TRAIN_QUANTITY = 5000
RZ_VALIDATION_QUANTITY = 1000
RZ_TEST_QUANTITY = 1000
RZ_MIN_ARTICLE_LENGTH = 200
