import json
import os
import re
from glob import glob

from scripts import settings

pattern_line = "\|"
pattern_category = '<META NAME="DZIAL" CONTENT="([, A-Ża-ż0-9\/\"\-]*)">'
pattern_p = "<[pP]?>"
pattern_meta = "<META NAME"
pattern_tag = "<[ _\-\.=a-zA-Z0-9/\"]+>"
pattern_n = "\n *"
pattern_blank = "  +"
categories = {}
os.makedirs(os.path.join(settings.DATA_DIR, "rz", "source"), exist_ok=True)

for file in glob(os.path.join(settings.DATA_DIR, "rz", "Rzeczpospolita", "*.html")):
    with open(file, "r", encoding="iso-8859-2") as source:
        source_text = source.read()
    file_name = file.split('\\')[-1].split('.')[0]

    if re.search(pattern_line, source_text):
        # print(f"line: {file_name}")
        continue

    result = re.search(pattern_category, source_text)
    try:
        category = source_text[result.regs[1][0]:result.regs[1][1]]
        if category in settings.RZ_TOPICS.keys():
            categories[file_name] = settings.RZ_TOPICS[category]
    except AttributeError:
        print(f"category: {file_name}")

    if os.path.exists(os.path.join(settings.DATA_DIR, "rz", "source", f"{file_name}.txt")):
        continue

    result = re.search(pattern_p, source_text)
    try:
        source_text = source_text[result.regs[0][1]:]
    except AttributeError:
        print(f"p: {file_name}")

    result = re.search(pattern_meta, source_text)
    try:
        source_text = source_text[:result.regs[0][0]]
    except AttributeError:
        print(f"meta: {file_name}")

    result = re.search(pattern_tag, source_text)
    while result:
        source_text = source_text[:result.regs[0][0]] + " " + source_text[result.regs[0][1]:]
        result = re.search(pattern_tag, source_text)

    result = re.search(pattern_n, source_text)
    while result:
        source_text = source_text[:result.regs[0][0]] + source_text[result.regs[0][1]:]
        result = re.search(pattern_n, source_text)

    result = re.search(pattern_blank, source_text)
    while result:
        source_text = source_text[:result.regs[0][0]] + source_text[result.regs[0][1] - 1:]
        result = re.search(pattern_blank, source_text)

    with open(os.path.join(settings.DATA_DIR, "rz", "source", f"{file_name}.txt"), "w", encoding="utf-8") as output:
        output.write(source_text)

with open(os.path.join(settings.RZ_DATA_DIR, settings.RZ_LABELS), "w", encoding="utf-8") as labels_file:
    json.dump(categories, labels_file)
