import os
import json
import sys

dir = sys.argv[1]

def to_jsonl(f, lang):
    lines = f.read().split("\n")
    del lines[-1]
    del lines[0]
    lines = [l.split("\t") for l in lines]
    dicts = [{"sentence1": l[0], "sentence2": l[1], "language": lang, "gold_label": l[2]} for l in lines]
    return dicts

for f in os.listdir(dir):
    if f.endswith(".tsv"):
        fr = open(f)
        prefix, split, lang, _ = f.split(".") 
        dicts = to_jsonl(fr, lang)
        with open(prefix + "." + split + "." + lang + ".jsonl", "w") as fw:
            for d in dicts:
                json.dump(d, fw)
                fw.write("\n")