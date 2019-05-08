from glob import glob
import os
from scripts import settings
import re


pattern_category = '<META NAME="DZIAL" CONTENT="([, A-Ża-ż0-9\/\"]*)">'
pattern_p = "<[pP]?>"
pattern_meta = "<META NAME"
pattern_tag = "<[ =a-zA-Z0-9/]+>"
pattern_n = "\n *"
pattern_blank = "  +"

categories = {}


for file in glob(os.path.join(settings.DATA_DIR, "rz", "Rzeczpospolita", "*.html")):
    with open(file, "r", encoding="iso-8859-2") as source:
        source_text = source.read()
    file_name = file.split('\\')[-1].split('.')[0]
    result = re.search(pattern_category, source_text)
    try:
        category = source_text[result.regs[1][0]:result.regs[1][1]]
        if category in categories.keys():
            categories[category] += 1
        else:
            categories[category] = 1
    except:
        pass

    result = re.search(pattern_p, source_text)
    try:
        source_text = source_text[result.regs[0][1]:]
    except:
        print(file_name)

    result = re.search(pattern_meta, source_text)
    source_text = source_text[:result.regs[0][0]]

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

print(categories)


