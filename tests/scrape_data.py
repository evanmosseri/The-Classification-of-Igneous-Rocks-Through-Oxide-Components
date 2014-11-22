import pandas as pd 
import numpy as np 
import itertools
from pprint import pprint

def prn(x): print x

# file_dir = "../data/unparsed_data.xlsx"
file_dir = "../data/peridotite_data_with_lattitude_longitude.csv"
# file_dir = "./cut_basalt.csv"
# handle = pd.read_csv(file_dir)
# file_dir = "./unparsed_data_v2.csv"
# handle = pd.read_excel(file_dir)
handle = pd.read_csv(file_dir)
handle = handle[handle["ROCK NAME"] != "PERIDOTITE"]
cols = handle.keys()[1:].tolist()

print handle.keys()


# print handle.dropna(axis=0)

comb = handle.keys()[3:]
combs = list(itertools.combinations(comb,6)) 
# combs = list(itertools.combinations(comb,4)) 


handles = sorted([[map(lambda x: str(x),components),len(handle[list(components)].dropna(axis=0))] for components in combs],key=lambda x:x[1],reverse=True)

# handles[0][0] = ['SIO2', 'CAO', 'AL2O3','MGO']

acronym = "".join(x[0] for x in handles[0][0])
# stam = handle[["ROCK NAME"]+list(components)].dropna(axis=0)
# stam = handle[["ROCK NAME"]+list(components)].dropna(axis=0)

# print len(stam)
# print stam
print acronym,handles[0][0],len(handle[handles[0][0]].dropna(axis=0))
handle[["ROCK NAME","LATITUDE","LONGITUDE"]+handles[0][0]].dropna(axis=0).to_csv("../data/new_"+acronym+".csv",index=False)
