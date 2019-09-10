In [1]: f = open("multinli.train.en.jsonl)                                                                                                                                             
  File "<ipython-input-1-8265554bc75f>", line 1
    f = open("multinli.train.en.jsonl)
                                      ^
SyntaxError: EOL while scanning string literal


In [2]: f = open("multinli.train.en.jsonl")                                                                                                                                            

In [3]: import json                                                                                                                                                                    

In [4]: for l in f: 
   ...:     e = json.loads(l) 
   ...:     s1 = e["sentence1]                                                                                                                                                                                                       
  File "<ipython-input-4-c5359682cb3f>", line 3
    s1 = e["sentence1]
                      ^
SyntaxError: EOL while scanning string literal


In [5]: for l in f: 
   ...:     e = json.loads(l) 
   ...:     s1 = e["sentence1"] 
   ...:     s2 = e["sentence2"] 
   ...:     s = s1 + " " + s2 
   ...:                                                                                                                                                                                                                              

In [6]: f = open("multinli.train.en.jsonl")                                                                                                                                                                                          

In [7]: lines = [] 
   ...: for l in f: 
   ...:     e = json.loads(l) 
   ...:     s1 = e["sentence1"] 
   ...:     s2 = e["sentence2"] 
   ...:     s = s1 + " " + s2 
   ...:     lines.append(l) 
   ...:                                                                                                                                                                                                                              

In [8]: len(lines)                                                                                                                                                                                                                   
Out[8]: 392702

In [9]: lines[2]                                                                                                                                                                                                                     
Out[9]: '{"sentence1": "One of our number will carry out your instructions minutely .", "sentence2": "A member of my team will execute your orders with immense precision .", "language": "en", "gold_label": "entailment"}\n'

In [10]: f = open("multinli.train.en.jsonl")                                                                                                                                                                                         

In [11]: lines = [] 
    ...: for l in f: 
    ...:     e = json.loads(l) 
    ...:     s1 = e["sentence1"] 
    ...:     s2 = e["sentence2"] 
    ...:     s = s1 + " " + s2 
    ...:     lines.append(s) 
    ...:      
    ...:                                                                                                                                                                                                                             

In [12]: len(lines)                                                                                                                                                                                                                  
Out[12]: 392702

In [13]: lines[2]                                                                                                                                                                                                                    
Out[13]: 'One of our number will carry out your instructions minutely . A member of my team will execute your orders with immense precision .'

In [14]: lines[2]                                                                                                                                                                                                                    
Out[14]: 'One of our number will carry out your instructions minutely . A member of my team will execute your orders with immense precision .'

In [15]: lines[3]                                                                                                                                                                                                                    
Out[15]: 'How do you know ? All this is their information again . This information belongs to them .'

In [16]: with open('lines_en.txt', mode='wt', encoding='utf-8') as myfile: 
    ...:     myfile.write('\n'.join(lines)) 
