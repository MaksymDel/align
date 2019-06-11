`cd data/mutlinli`
`wget http://www.nyu.edu/projects/bowman/multinli/multinli_1.0.zip`
`unzip multinli_1.0.zip`
`mv multinli_1.0/* .`

`cd ../../data/xnli`
`wget https://www.nyu.edu/projects/bowman/xnli/XNLI-1.0.zip`
`unzip XNLI-1.0.zip`
`mv XNLI-1.0/* .`
`split xnli.dev.jsonl xnli.dev. -a 1 -l 2490`
`split xnli.test.jsonl xnli.test. -a 1 -l 5010`
`python rename_files.py`