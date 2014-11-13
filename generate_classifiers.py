""" TODO:
create binary classification
"""
import numpy as np
import pandas as pd
import scipy as st
from rocksep_utils import *
import pickle
from sklearn.externals import joblib
import itertools

cmap = {1:"red",2:"green",3:"blue",4:"orange"}
plot_vars = ["SIO2","AL2O3","MGO"]

algorithms = [nearest_neighbors,random_forest,naive_base,ada,lda,qda,decision_tree,svc]

# for x in random_data:
	# print x

for x in random_data:
	print x
x = plot_algorithms(algorithms,raw_data,nlabels,plot_vars,save=False,pl=True,mat=True,plot=True,pca=False,directory="./figures/scam_save/",suppress=False)
# x = plot_algorithms(algorithms,raw_data,nlabels,plot_vars,save=False,pl=True,mat=True,plot=False,pca=True,directory="./figures/test")
# joblib.dump(x['decision_tree'][5], 'decision_tree_model.pkl', compress=9)

# classifier = joblib.load(open("./figures/classifiers/nearest_neighbors.pkl"))
# print x['decision_tree'][5]
# x = joblib.load("test.pkl")	

# for i in range(len(x)):
	# joblib.dump(x['decision_tree'][5], open("./figures/classifiers/"+x.keys()[i]+".pkl"), compress=9)	
	# joblib.dump(x[x.keys()[i]][5], "./figures/classifier_app/{}.pkl".format(x.keys()[i]), compress=9)
