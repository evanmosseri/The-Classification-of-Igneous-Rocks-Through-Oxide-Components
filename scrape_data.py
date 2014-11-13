import pandas as pd 
import numpy as np 
import itertools
from pprint import pprint

def prn(x): print x

file_dir = "./data/unparsed_data.xlsx"
# file_dir = "./cut_basalt.csv"
# handle = pd.read_csv(file_dir)
# file_dir = "./unparsed_data_v2.csv"
handle = pd.read_excel(file_dir)
cols = handle.keys()[1:].tolist()

print handle.keys()


# print handle.dropna(axis=0)

comb = handle.keys()[1:]
combs = list(itertools.combinations(comb,6)) 


handles = sorted([[map(lambda x: str(x),components),len(handle[list(components)].dropna(axis=0))] for components in combs],key=lambda x:x[1],reverse=True)


acronym = "".join(x[0] for x in handles[0][0])

# print generate_acronym(handles[1][0])

# components = combs[np.argmax(handles)]
# print(*handles,sep="\n")

# map(prn,handles)
# pprint(handles)


# print components
# print len(handle[list(components)])

# for i in range(len(combs)):
	# print combs[i],handles[i]










# print len(handles[components])
# handle = handle[list(components)].dropna(axis=0)
# print len(handle.values)

# print comb[0:4].tolist(),comb[5]
# print comb[0:4]+[comb[5]]


# component = ["SiO2","TiO2", "Al2O3", "Fe2O3", "Fe2O3T", "FeO", "FeOT", "NiO", "MnO", "MgO", "CaO"]
# print handle
# print handle
# print comb
# print np.unique(handle["Material"].values)
# handle=handle[["Material"]+comb[0:4].tolist()+comb[8:11].tolist()].dropna(axis=0)
# handle = handle[component]

# print handle

# components = ["Material","SIO2","TIO2","AL2O3","CAO"]
# components = ["ROCK NAME","SIO2","CR2O3","FEOT","MNO"]






# stam = handle[["ROCK NAME"]+list(components)].dropna(axis=0)
# stam = handle[["ROCK NAME"]+list(components)].dropna(axis=0)

# print len(stam)
# print stam
# print acronym,handles[0][0],len(handle[handles[0][0]].dropna(axis=0))
print len(handle)
# handle[["ROCK NAME"]+handles[0][0]].dropna(axis=0).to_csv(acronym+".csv",index=False)


