## DATASET OVERLAP IN GZIP PAPER ##

from data import *
from datasets import load_dataset
from tabulate import tabulate

ds_info = [

    #("filipino",
    #['dengue_filipino'],
    #['text', 'absent', 'dengue', 'health', 'mosquito', 'sick']),

    # Train and test set was 100% identical at filipino dataset

    ("kirnews",
     ["kinnews_kirnews","kirnews_cleaned"],
     ['label','title','content']),
    
    ("kinnews",
     ["kinnews_kirnews", "kinnews_cleaned"],
     ['label','title','content']),

    ("swahili",
     ['swahili_news'],
     ['label','text']),

]

lines = []
for name,args,keys in ds_info:

    ds = load_dataset(*args)

    # convert to list-of-tuples:
    train = [tuple([item[key] for key in keys]) for item in ds['train']]
    test  = [tuple([item[key] for key in keys]) for item in ds['test']]

    lines.append(name)

    n_overlap = len(set(train).intersection(test))
    lines.append(tabulate([
        ("train:",        len(train)),
        ("train unique:", len(set(train))),
        ("test:",         len(test)),
        ("test unique:",  len(set(test))),
        
        ("train/test overlap:", n_overlap,
         "%.1f%%" % (100.0 * n_overlap / len(set(test)))),
    ]))
    lines.append("\n")
    
print("\n".join(lines))