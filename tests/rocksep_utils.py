from StringIO import StringIO
import time
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mpld3
import numpy as np
import pandas as pd
from pydot import graph_from_dot_data
from rocksep_utils import *
import rocksep_utils as utl
import scipy as st
from sklearn import cross_validation
from sklearn.cross_validation import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.externals import joblib
from sklearn.lda import LDA
from sklearn.lda import LDA
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.qda import QDA
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import sklearn.tree as tree


# remember to change this when changing random data
# dataURL = "./data/peridotites_clean_complete.csv"
dataURL = "./data/peridotites_clean_SCAM.csv"
# dataURL = "new.csv"
# now I have a problem with split and c_validation when using all data
v_type = ["split","c_validation","k_fold"][2]
rock_types={'DUNITE' :1,'HARZBURGITE':2, 'LHERZOLITE':3, 'WEHRLITE':4}

pd_data = pd.read_csv(dataURL)
chemicals = pd_data.keys()[1:].tolist()
rock_names = np.unique(pd_data[pd_data.keys()[0]]).tolist()
scam = ["SIO2","CAO","AL2O3","MGO"]



random_data = pd_data.values
np.random.shuffle(random_data)


# np.save("random_data_scam.npy",random_data)
# random_data = np.load("./data/random_data_SCAM.npy")
# random_data = np.load("./data/random_data.npy")
# random_data = np.load("./data/new_data.npy")
raw_data = random_data[:,1:]
labels = random_data[:,0]
# raw_data = random_data.iloc[:,1:].values
# labels = random_data.iloc[:,0].values
# mask = np.isfinite(raw_data).all(axis=1)
# raw_data = raw_data[mask]
# labels = labels[mask]

print chemicals,raw_data
def numbered_labels(labels,key):
	return [key[labels[i]] for i in range(len(labels))]
nlabels =  numbered_labels(labels,rock_types)

def s_k_fold(feature_mat,ylabels):
	train, test = iter(StratifiedKFold(ylabels, n_folds=4)).next()
	feature_train, feature_test = [feature_mat[x] for x in train], [feature_mat[x] for x in test]
	ylabels_train, ylabels_test = [ylabels[x] for x in train], [ylabels[x] for x in test]
	X,y = feature_train,ylabels_train
	X_test,y_test = feature_test,ylabels_test
	return np.array(X),np.array(y),np.array(X_test),np.array(y_test)

# this one uses the gloabl labels, should eventually allow individual label analysis
def split_data(raw_data,labels,type):
	if v_type == "split":
		train_data,test_data,train_labels,test_labels = raw_data[0:int(len(raw_data)/2)],raw_data[int(len(raw_data)/2):],nlabels[0:int(len(nlabels)/2)],nlabels[int(len(nlabels)/2):]
	elif v_type == "c_validation":
		train_data,test_data,train_labels,test_labels = cross_validation.train_test_split(raw_data, nlabels, test_size=0.3, random_state=0)
	elif v_type =="k_fold":
		train_data,train_labels,test_data,test_labels =  s_k_fold(raw_data,nlabels)
	print "{} training examples".format(len(train_data))
	print "{} testing examples".format(len(test_data))
	return train_data,train_labels,test_data,test_labels

def nearest_neighbors(data,labels,n,v_type):
	train_data,train_labels,test_data,test_labels = split_data(data,labels,v_type)

	clf = KNeighborsClassifier(n_neighbors=n)
	clf.fit(train_data, train_labels)
	y_pred = clf.predict(test_data)
	pure_accuracy_rate = len([y_pred[x] for x in range(len(y_pred)) if y_pred[x] == test_labels[x]])/float(len(test_labels))
	report = classification_report(y_pred, test_labels, target_names=rock_names)
	cm = confusion_matrix(test_labels, y_pred)
	return pure_accuracy_rate,report,y_pred,test_labels,test_data,clf,cm,"Nearest Neighbors"

def random_forest(data,labels,n,v_type):
	train_data,train_labels,test_data,test_labels = split_data(data,labels,v_type)

	clf = RandomForestClassifier(n_estimators=n)
	clf.fit(train_data, train_labels)
	y_pred = clf.predict(test_data)
	pure_accuracy_rate = len([y_pred[x] for x in range(len(y_pred)) if y_pred[x] == test_labels[x]])/float(len(test_labels))
	report = classification_report(y_pred, test_labels, target_names=rock_names)
	cm = confusion_matrix(test_labels, y_pred)
	return pure_accuracy_rate,report,y_pred,test_labels,test_data,clf,cm,"Random Forest"

def naive_base(data,labels,n,v_type):
	train_data,train_labels,test_data,test_labels = split_data(data,labels,v_type)

	clf = GaussianNB()
	clf.fit(train_data, train_labels)
	y_pred = clf.predict(test_data)
	pure_accuracy_rate = len([y_pred[x] for x in range(len(y_pred)) if y_pred[x] == test_labels[x]])/float(len(test_labels))
	report = classification_report(y_pred, test_labels, target_names=rock_names)
	cm = confusion_matrix(test_labels, y_pred)
	return pure_accuracy_rate,report,y_pred,test_labels,test_data,clf,cm,"Naive Base"

def qda(data,labels,n,v_type):
	train_data,train_labels,test_data,test_labels = split_data(data,labels,v_type)

	clf = QDA(priors=None, reg_param=0.0)
	clf.fit(train_data, train_labels)
	y_pred = clf.predict(test_data)
	pure_accuracy_rate = len([y_pred[x] for x in range(len(y_pred)) if y_pred[x] == test_labels[x]])/float(len(test_labels))
	report = classification_report(y_pred, test_labels, target_names=rock_names)
	cm = confusion_matrix(test_labels, y_pred)
	return pure_accuracy_rate,report,y_pred,test_labels,test_data,clf,cm,"QDA"

def svc(data,labels,n,v_type):
	train_data,train_labels,test_data,test_labels = split_data(data,labels,v_type)

	clf = SVC(kernel="linear", C=0.025)
	clf.fit(train_data, train_labels)
	y_pred = clf.predict(test_data)
	pure_accuracy_rate = len([y_pred[x] for x in range(len(y_pred)) if y_pred[x] == test_labels[x]])/float(len(test_labels))
	report = classification_report(y_pred, test_labels, target_names=rock_names)
	cm = confusion_matrix(test_labels, y_pred)
	return pure_accuracy_rate,report,y_pred,test_labels,test_data,clf,cm,"SVC"

def lda(data,labels,n,v_type):
	train_data,train_labels,test_data,test_labels = split_data(data,labels,v_type)

	clf = LDA()
	clf.fit(np.array(train_data,dtype=np.float64), np.array(train_labels,dtype=np.float64))
	y_pred = clf.predict(test_data)
	pure_accuracy_rate = len([y_pred[x] for x in range(len(y_pred)) if y_pred[x] == test_labels[x]])/float(len(test_labels))
	report = classification_report(y_pred, test_labels, target_names=rock_names)
	cm = confusion_matrix(test_labels, y_pred)
	return pure_accuracy_rate,report,y_pred,test_labels,test_data,clf,cm,"LDA"

def qda(data,labels,n,v_type):
	train_data,train_labels,test_data,test_labels = split_data(data,labels,v_type)

	clf = QDA()
	clf.fit(train_data, train_labels)
	y_pred = clf.predict(test_data)
	pure_accuracy_rate = len([y_pred[x] for x in range(len(y_pred)) if y_pred[x] == test_labels[x]])/float(len(test_labels))
	report = classification_report(y_pred, test_labels, target_names=rock_names)
	cm = confusion_matrix(test_labels, y_pred)
	return pure_accuracy_rate,report,y_pred,test_labels,test_data,clf,cm,"QDA"


def ada(data,labels,n,v_type):
	train_data,train_labels,test_data,test_labels = split_data(data,labels,v_type)

	clf = AdaBoostClassifier()
	clf.fit(train_data, train_labels)
	y_pred = clf.predict(test_data)
	pure_accuracy_rate = len([y_pred[x] for x in range(len(y_pred)) if y_pred[x] == test_labels[x]])/float(len(test_labels))
	report = classification_report(y_pred, test_labels, target_names=rock_names)
	cm = confusion_matrix(test_labels, y_pred)
	return pure_accuracy_rate,report,y_pred,test_labels,test_data,clf,cm,"ADA Boost"  

def decision_tree(data,labels,n,v_type):
	train_data,train_labels,test_data,test_labels = split_data(data,labels,v_type)

	clf = DecisionTreeClassifier(max_depth=5)
	clf.fit(train_data, train_labels)
	y_pred = clf.predict(test_data)
	pure_accuracy_rate = len([y_pred[x] for x in range(len(y_pred)) if y_pred[x] == test_labels[x]])/float(len(test_labels))
	report = classification_report(y_pred, test_labels, target_names=rock_names)
	cm = confusion_matrix(test_labels, y_pred)
	return pure_accuracy_rate,report,y_pred,test_labels,test_data,clf,cm,"Decision Tree" 


def plotAnalysis(test_data,test_labels,y_pred,cmap,plot_vars,type,title,accuracy,plot=True,cache=False,save=False,analysis_type="",directory="./matplotlib_save/"):
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	x1,y1,z1 = [test_data[:,chemicals.index(x)] for x in plot_vars]
	hit_shapes = {True:"*",False:"o"}

	hit_mat = [y_pred[i] == test_labels[i] for i in range(len(test_labels))]
	hit_shapes_mat = [hit_shapes[i] for i in hit_mat]
	rock_colors = [cmap[x] for x in test_labels]

	for i in range(len(hit_shapes_mat)):
		ax.scatter(x1[i], y1[i], zs=z1[i], c=rock_colors[i], marker=hit_shapes_mat[i],s=50)

	proxy_labels = [matplotlib.lines.Line2D([0],[0], linestyle="none", c='black', marker = 'o'), matplotlib.lines.Line2D([0],[0], linestyle="none", c='black', marker = '*')]
	for i in range(1,5):
		proxy_labels.append(matplotlib.lines.Line2D([0],[0], linestyle="none", c=cmap[i], marker = 'o'))
	ax.legend(proxy_labels, ['Incorrectly Classified', 'Correctly Classified']+rock_names, numpoints = 1,prop={'size':6},bbox_to_anchor=(1.1, .1))
	ax.set_xlabel(plot_vars[0])
	ax.set_ylabel(plot_vars[1])
	ax.set_zlabel(plot_vars[2])
	ax.view_init(elev=57,azim=82)
	ax.set_title(analysis_type+": "+title+":\n Overall Accuracy: "+"%.1f%%" % accuracy)
	if plot == True:
		plt.show()
	if cache == True:
		plt.savefig("./matplotlib_cache/{}_figure{}.png".format(analysis_type,time.time()))
	if save == True:
		plt.savefig("{}/{}_figure{}.png".format(directory,analysis_type,time.time())) 
	return fig
# might not be correct 
def plotAnalysisPCA(test_data,test_labels,y_pred,cmap,plot_vars,type,title,accuracy,plot=True,cache=False,save=False,analysis_type="",directory="./matplotlib_save"):
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	# x1,y1,z1 = [test_data[:,chemicals.index(x)] for x in plot_vars]

	hit_shapes = {True:"*",False:"o"}
	pca = PCA(n_components=3)
	x = pca.fit(test_data)
	raw_data_reduced = PCA.transform(x,test_data)
	x1,y1,z1 = [raw_data_reduced[:,x] for x in range(len(raw_data_reduced[0]))]
	r_samples = np.array([[np.array([x1[i],y1[i],z1[i],test_labels[i],y_pred[i]]) for i in range(len(test_labels)) if test_labels[i] == c] for c in range(1,5) ] ) 

	hit_mat = [y_pred[i] == test_labels[i] for i in range(len(test_labels))]
	hit_shapes_mat = [hit_shapes[i] for i in hit_mat]
	rock_colors = [cmap[x] for x in test_labels]
	for x in range(len(r_samples)):
		samples = [[i for i in r_samples[x] if i[3] == i[4]],[i for i in r_samples[x] if i[3] != i[4]]]
		# print samples[0]
		ax.scatter(np.array(samples[0])[:,0],np.array(samples[0])[:,1],zs=np.array(samples[0])[:,2],c=cmap[x+1],marker = '*')
		ax.scatter(np.array(samples[1])[:,0],np.array(samples[1])[:,1],zs=np.array(samples[1])[:,2],c=cmap[x+1],marker = 'o')
	# for i in range(len(hit_shapes_mat)):
	# 	ax.scatter(x1[i], y1[i], zs=z1[i], c=rock_colors[i], marker=hit_shapes_mat[i],s=50)

	proxy_labels = [matplotlib.lines.Line2D([0],[0], linestyle="none", c='black', marker = 'o'), matplotlib.lines.Line2D([0],[0], linestyle="none", c='black', marker = '*')]
	for i in rock_types.values():
		proxy_labels.append(matplotlib.lines.Line2D([0],[0], linestyle="none", c=cmap[i], marker = 'o'))
	ax.legend(proxy_labels, ['Incorrectly Classified', 'Correctly Classified']+rock_names, numpoints = 1,prop={'size':6},bbox_to_anchor=(1.1, .1))
	ax.set_xlabel("PCA Component 0")
	ax.set_ylabel("PCA Component 1")
	ax.set_zlabel("PCA Component 2")
	ax.set_title(analysis_type+": "+title+":\n Overall Accuracy: "+"%.1f%%" % accuracy)
	ax.view_init(elev=57,azim=82)
	if plot == True:
		plt.show()
	if cache == True:
		plt.savefig("./matplotlib_cache/pca_{}_figure{}.png".format(analysis_type,time.time()))
	if save == True:
		plt.savefig("{}/pca_{}_figure{}.png".format(directory,analysis_type,time.time())) 
	return fig
def exportDecisionTree(clf,filename):
	out = StringIO()
	out = tree.export_graphviz(classifier, out_file=out)
	graph_from_dot_data(out.getvalue()).write_pdf(filename)
	return out
def show_confusion_matrix(cm,plot,cache,save,a_type="",accuracy=None,directory="./matplotlib_save/"):
	fig = plt.figure()
	ax0 = fig.add_subplot(1,1,1)
	def normalize_mat(cm):
		cm = cm.tolist()
		for i in range(len(cm)):
			rsum = np.sum(cm[i])
			for x in range(len(cm[0])):
				cm[i][x] = float(cm[i][x])/rsum
		return np.array(cm)
	im = ax0.imshow(normalize_mat(cm), interpolation='nearest',cmap='Reds')
	plt.xticks(np.arange(0,4), rock_names)
	plt.yticks(np.arange(0,4), rock_names)
	plt.colorbar(im)
	ax0.set_title("{} Confusion Matrix\n {} accuracy".format(a_type,accuracy))
	if plot == True:
		plt.show()
	if cache == True:
		plt.savefig("./matplotlib_cache/{}confusion_matrix{}.png".format(a_type,time.time()))
	if save == True:
		plt.savefig("{}/{}confusion_matrix{}.png".format(directory,a_type,time.time())) 
	return fig
def plot_algorithms(algorithms,raw_data,nlabels,plot_vars,cmap={1:"red",2:"green",3:"blue",4:"orange"},save=False,plot=False,mat=True,pl=True,pca=False,directory="./matplotlib_save",suppress=False):
	values = {}
	for i in algorithms:
		pure_accuracy_rate,report,y_pred,test_labels,test_data,clf,cm,analysis_type = i(raw_data,nlabels,4,v_type=v_type)
		values[i.func_name] = [pure_accuracy_rate,report,y_pred,test_labels,test_data,clf,cm,analysis_type];
		pure_accuracy_rate = pure_accuracy_rate*100
		if not(suppress):
			print analysis_type+"\n","Simple Accuracy: %.1f%%" % (pure_accuracy_rate)+"\n", report
		if save:
			if pl:
				if pca:
					fig = plotAnalysisPCA(test_data,test_labels,y_pred,cmap,plot_vars,analysis_type,"Principle Component Analysis of Classifier Accuracy",pure_accuracy_rate,plot=False,cache=False,save=True,analysis_type=i.func_name.title().replace("_"," "),)
				else:	
					fig = plotAnalysis(test_data,test_labels,y_pred,cmap,plot_vars,analysis_type,"3 Dimensional Demonstration of Classifier Accuracy",pure_accuracy_rate,plot=False,cache=False,save=True,analysis_type=i.func_name.title().replace("_"," "),directory=directory)
			if mat:
				show_confusion_matrix(cm,False,False,True,a_type=i.func_name.title().replace("_"," "),accuracy="%.1f%%" % (pure_accuracy_rate),directory=directory) 
		elif plot:
			if pl:
				if pca:
					fig = plotAnalysisPCA(test_data,test_labels,y_pred,cmap,plot_vars,analysis_type,"Principle Component Analysis of Classifier Accuracy",pure_accuracy_rate,plot=True,cache=False,save=False,analysis_type=i.func_name.title().replace("_"," "),directory=directory)
				else:
					fig = plotAnalysis(test_data,test_labels,y_pred,cmap,plot_vars,analysis_type,"3 Dimensional Demonstration of Classifier Accuracy",pure_accuracy_rate,plot=True,cache=False,save=False,analysis_type=i.func_name.title().replace("_"," "))
			if mat:
				show_confusion_matrix(cm,True,False,False,a_type=i.func_name.title().replace("_"," "),accuracy="%.1f%%" % (pure_accuracy_rate)) 
	return values

