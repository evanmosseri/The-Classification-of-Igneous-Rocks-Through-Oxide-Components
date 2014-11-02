import argparse
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

parser = argparse.ArgumentParser()
parser.add_argument("SCAM",help="[SIO2,CAO,AL2O3,MGO]",type=str)
parser.add_argument("-classifier",help="optional: specify classifier file",type=str,default="nearest_neighbors.pkl")
parser.add_argument("-buffer",help="specify which column your SCAM values starts at (default: 0)",type=int,default=0)
parser.add_argument("-sheet",help="excel sheet to use (required if file is xlsx)",type=str)
args = parser.parse_args()

rock_types={'DUNITE' :1,'HARZBURGITE':2, 'LHERZOLITE':3, 'WEHRLITE':4}
num_to_rock = {v: k for k, v in rock_types.items()}

def list_to_rocks(labels):
	return [num_to_rock[x] if x<5 else "error" for x in labels]


test_array = []
dataset = None
prediction = []
filetype = ""

if ".csv" in args.SCAM:
	filetype = "csv"
	if os.path.isfile(args.SCAM):
		dataset = args.SCAM[0:-4]
	else:
		sys.exit("file does not exist")
elif ".xlsx" in args.SCAM:
	filetype = "xlsx"
	if os.path.isfile(args.SCAM):
		dataset = args.SCAM[0:-5]
	else:
		sys.exit("file does not exist")
elif ".xls" in args.SCAM:
	filetype = "xls"
	if os.path.isfile(args.SCAM):
		dataset = args.SCAM[0:-5]
	else:
		sys.exit("file does not exist")
elif ("[" in args.SCAM) & ("]" in args.SCAM):
	test_array = [float(x) for x in args.SCAM.strip()[1:-1].split(',')]
else:
	sys.exit("Invalid Dataset")
if ".pkl" in args.classifier:
	if os.path.isfile("./classifiers/"+args.classifier):
		classifier_name = "./classifiers/"+args.classifier
classifier = joblib.load("./{}".format(classifier_name))
if len(test_array) == 4:
	print list_to_rocks(classifier.predict(test_array))
if dataset != None: 
	if filetype == "csv":
		dataframe = pd.read_csv(dataset+".csv")
	elif filetype == "xlsx":
		dataframe = pd.read_excel(dataset+".xlsx", args.sheet, index_col=None, na_values=['NA'])
	elif filetype == "xls":
		dataframe = pd.read_excel(dataset+".xlsx", args.sheet, index_col=None, na_values=['NA'])
	vals = dataframe[dataframe.columns[0+args.buffer:4+args.buffer]].values.tolist()
	prediction = list_to_rocks(classifier.predict(vals))
	dataframe['Classification'] = [x.title() for x in prediction]
	if filetype == "csv":
		dataframe.to_csv(dataset+"_classified.csv")
	elif filetype == "xlsx":
		dataframe.to_excel(dataset+"_classified.xlsx",args.sheet)
	elif filetype == "xls":
		dataframe.to_excel(dataset+"_classified.xls",args.sheet)




