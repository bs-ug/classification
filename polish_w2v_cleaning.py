import os
import re

pattern_pos = "::[a-z]+"
pattern_ne = "ne#"

with open(os.path.join("..", "data", "models", "skip_gram_v100m8.w2v"), "r", encoding="utf-8") as input_file, \
        open(os.path.join("..", "data", "models", "polish_100.w2v"), "w", encoding="utf-8") as output_file:
    for line in input_file:
        try:
            result = re.search(pattern_pos, line)
            line = line[:result.regs[0][0]] + line[result.regs[0][1]:]
        except AttributeError:
            pass
        try:
            result = re.search(pattern_ne, line)
            line = line[:result.regs[0][0]] + line[result.regs[0][1]:]
        except AttributeError:
            pass
        output_file.write(line)
