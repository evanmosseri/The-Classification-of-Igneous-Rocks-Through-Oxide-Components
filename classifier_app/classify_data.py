from sklearn.externals import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.lda import LDA
from sklearn.qda import QDA
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import os.path
import pandas as pd
import sys

rock_types={'DUNITE' :1,'HARZBURGITE':2, 'LHERZOLITE':3, 'WEHRLITE':4}
num_to_rock = {v: k for k, v in rock_types.items()}

def list_to_rocks(labels):
	return [num_to_rock[x] if x<5 else "error" for x in labels]


test_array = []
dataset = None
classifier_name = "./classifiers/nearest_neighbors.pkl"
prediction = []
if len(sys.argv) > 1:
	if ".csv" in sys.argv[1]:
		if os.path.isfile(sys.argv[1]) & (".csv" in sys.argv[1]):
			dataset = sys.argv[1][0:-4]
		else:
			sys.exit("file does not exist")
	elif "[" in sys.argv[1]:
		test_array = [float(x) for x in sys.argv[1].strip()[1:-1].split(',')]
	if len(sys.argv) > 2:
		if ".pkl" in sys.argv[2]:
			if os.path.isfile(sys.argv[2]):
				classifier_name = "./classifiers/"+sys.argv[2]
classifier = joblib.load("./{}".format(classifier_name))
if len(test_array) == 4:
	print list_to_rocks(classifier.predict(test_array))
if dataset != None: 
	dataframe = pd.read_csv(dataset+".csv")
	vals = dataframe[dataframe.columns[0:4]].values.tolist()
	prediction = list_to_rocks(classifier.predict(vals))
	dataframe['Classified'] = prediction 
	dataframe.to_csv(dataset+"_classified.csv")