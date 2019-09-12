import json                                                                                                                                                                    
                                                                                                                                                                                                                              
f = open("multinli.train.de.jsonl")                                                                                                                                                                                          

lines = []
for l in f: 
  e = json.loads(l) 
  s1 = e["sentence1"] 
  s2 = e["sentence2"] 
  s = s1 + " " + s2 
  lines.append(s)                                                                                                                                                                                                                               

with open('lines_de.txt', mode='wt', encoding='utf-8') as myfile: 
  myfile.write('\n'.join(lines)) 
