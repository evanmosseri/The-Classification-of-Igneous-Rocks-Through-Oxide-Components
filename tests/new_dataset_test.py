from StringIO import StringIO
import time
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mpld3
import numpy as np
import pandas as pd
from pydot import graph_from_dot_data
import scipy as st
from sklearn import cross_validation
from sklearn.cross_validation import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.externals import joblib
from sklearn.lda import LDA
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.qda import QDA
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import sklearn.tree as tree

rockfiles = {'dunite':'./data/dunites_clean_SCAM.csv',
              'harzburgite':'./data/harz_matlab_SCAM.csv',
              'wehrlite':'./data/wehrl_matlab_SCAM.csv',
              'lherzolite':'./data/lherzolite_clean_SCAM.csv'
              }
labels = []
stacks = [pd.read_csv(f).values[:,1:] for f in rockfiles.values()] 
for i in range(len(stacks)):
	labels+=[rockfiles.keys()[i].upper()]*len(stacks[i])

merged_data = np.vstack(stacks)
labels = np.hstack(labels)[:,np.newaxis]
print labels.shape
merged_data = np.hstack((labels,merged_data))
new_arr = []
for i in range(len(merged_data)):
	for x in range(1,len(merged_data[0])):
		new_arr.append([merged_data[i,0]]+[float(merged_data[i,x]) for x in range(1,len(merged_data[0]))])
# print merged_data
# print new_arr
x = pd.DataFrame(new_arr).values
print x
# print np.array(x,dtype='float64')

# print np.array(new_arr)
# print np.array(new_arr,dtype=[('v',str),('w','float64'),('x','float64'),('y', 'float64'), ('z', 'float64')])
# np.random.shuffle(merged_data)
# print merged_data
# np.save("./data/new_data.npy",x)
# print np.isfinite(merged_data[:,1:]).all()








