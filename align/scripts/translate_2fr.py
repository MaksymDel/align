import json                                                                                                                                                                    
import torch
import sys

n = sys.argv[1]

f = open(n)                                                                                                                                                                                          

f_en = open(n + '.en', mode='wt', encoding='utf-8')
f_fr = open(n + '.fr', mode='wt', encoding='utf-8')

# Load an En-Fr Transformer model trained on WMT'14 data :
en2fr = torch.hub.load('pytorch/fairseq', 'transformer.wmt14.en-fr', tokenizer='moses', bpe='subword_nmt')
# Use the GPU (optional):
en2fr.cuda()


lines = []
en_dicts = []
fr_dicts = []

i = 0
for l in f: 
  i += 1
  if i % 1000 == 0:
    print(i) 
  e = json.loads(l) 
  s1 = e["sentence1"] 
  s2 = e["sentence2"]
  label = e["gold_label"]
  
  en_dict = {"sentence1": s1, "sentence2": s2, "language": "en", "gold_label": label}
  
  json.dump(en_dict, f_en)
  f_en.write("\n")                                                                                                                                                                                                                             

  s1_fr = en2fr.translate(s1, beam=5)
  s2_fr = en2fr.translate(s2, beam=5)
  fr_dict = en_dict = {"sentence1": s1_fr, "sentence2": s2_fr, "language": "fr", "gold_label": label}
  
  json.dump(fr_dict, f_fr)
  f_fr.write("\n")                                                                                                                                                                                                                             