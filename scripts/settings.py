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
TRAIN_QUANTITY = 2500
VALIDATION_QUANTITY = 100
TEST_QUANTITY = 100
CNN_TRAINING_FILES_PATH = "../data/cnn/train"
CNN_VALIDATION_FILES_PATH = "../data/cnn/validation"
CNN_TEST_FILES_PATH = "../data/cnn/test"
CNN_MODEL_PATH = "../data/cnn/model"
CNN_MODEL_NAME = "cnn_w2v.model"
