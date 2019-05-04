import os
from glob import glob
from scripts import settings
import re


pattern_photo = "Zdjęcie([\w ]*/[\w ]*/ " \
                "((East News)|(Reporter)|(PAP)|(PAP/EPA)|(AFP)|(Archiwum)|(123RF/PICSEL)|(RMF)|(FORUM)|" \
                "(Agencja FORUM)|(Getty Images Video: Reuters Archive)))|([\w ]*/ Grafika RMF FM /)"
pattern_br = "(<br/> *)|(<br> *)|(\\\\<br\\\\> *)"
pattern_n = "\n *"
pattern_blank = "  +"
pattern_error = "�"

counter = 0
for file in glob(os.path.join(settings.POLISH_TRAINING_FILES_PATH, "*.txt")):
    with open(file, "r", encoding="utf-8") as input_file:
        content = input_file.read()

    if re.search(pattern_error, content):
        continue
    #
    # result = re.search(pattern_br, content)
    # while result:
    #     start = result.regs[0][0]
    #     stop = result.regs[0][1]
    #     content = content[:start] + content[stop:]
    #     result = re.search(pattern_br, content)
    #
    # result = re.search(pattern_n, content)
    # while result:
    #     start = result.regs[0][0]
    #     stop = result.regs[0][1]
    #     content = content[:start] + content[stop:]
    #     result = re.search(pattern_n, content)
    #
    # result = re.search(pattern_blank, content)
    # while result:
    #     start = result.regs[0][0]
    #     stop = result.regs[0][1]
    #     content = content[:start] + content[stop - 1:]
    #     result = re.search(pattern_blank, content)

    result = re.search(pattern_photo, content)
    if result:
        counter += 1
        print(f"{counter}. {content[result.regs[0][0]:result.regs[0][1]+50]}")
