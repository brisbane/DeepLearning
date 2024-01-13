#!/usr/bin/env python
import pandas 
#dest_folder=self.dest_folder
from fairlib import datasets
from pathlib import Path
#already done so commented out
datasets.prepare_dataset("bios", "/home/user/miniconda3/data/bios")
dest_folder="/home/user/miniconda3/data/bios"
all=[]
splits=[('train', 9) , ('dev', 2) , ('test', 4) ]
for split, splitn in splits:
    for i in range(splitn):
       p=pandas.read_pickle(open("/home/user/miniconda3/data/bios/bios_{}_df_30000_{}.pkl".format(split, i),"rb"))
       all.append(p)
    df=pandas.concat(all)
    df.to_pickle(Path(dest_folder) / "bios_{}_df.pkl".format(split))
