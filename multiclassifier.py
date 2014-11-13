"""
Copyright (C) 2014 Evan Mosseri

This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as 
published by the Free Software Foundation; either version 2 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of 
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details. 
You should have received a copy of the GNU General Public License along with this program; 
if not, write to the Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA

"""

__author__ = "Evan Mosseri"
"""
	TODO:

"""

from StringIO import StringIO
import itertools
import math
import os
from pprint import pprint
import sys
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
from sklearn.cluster import KMeans
from sklearn.cross_validation import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.externals import joblib
from sklearn.lda import LDA
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier, PassiveAggressiveClassifier, SGDClassifier
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier,RadiusNeighborsClassifier
from sklearn.qda import QDA
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import sklearn.tree as tree
import pylab

class RockClassifier:
	@staticmethod
	def numbered_labels(labels,key):
		return [key[labels[i]] for i in range(len(labels))]
	@staticmethod
	def labels_to_keys(labels):
		return {labels[i]:i for i in range(len(labels))}
	@staticmethod
	def feed(csv,**kwargs):
		stam = pd.read_csv(csv)
		temp_data = stam.values
		np.random.shuffle(temp_data)
		label = temp_data[:,0]
		dat = temp_data[:,1:]
		chemicals = stam.keys()[1:]
		RockClassifier.classify.chemicals = chemicals
		try:
			RockClassifier.classify.rock_names = np.unique(stam["Rock Type"].values)
		except:
			RockClassifier.classify.rock_names = np.unique(stam[stam.keys()[0]].values)
		RockClassifier.classify.rock_types = RockClassifier.labels_to_keys(RockClassifier.classify.rock_names)
		return RockClassifier(np.array(dat,dtype=np.float64),RockClassifier.numbered_labels(label,RockClassifier.classify.rock_types),**kwargs)

	@staticmethod
	def file_acronym(filename):
		os.path.basename(filename)[0:0-len(os.path.splitext(filename)[1])]

	def s_k_fold(self,feature_mat,ylabels,n_folds=4):
		train, test = iter(StratifiedKFold(ylabels, n_folds=n_folds)).next()
		feature_train, feature_test = [feature_mat[x] for x in train], [feature_mat[x] for x in test]
		ylabels_train, ylabels_test = [ylabels[x] for x in train], [ylabels[x] for x in test]
		X,y = feature_train,ylabels_train
		X_test,y_test = feature_test,ylabels_test
		return np.array(X),np.array(y),np.array(X_test),np.array(y_test)
	def split_data(self,v_type="k_fold",n_folds=4):
			if v_type == "split":
				train_data,test_data,train_labels,test_labels = self.data[0:int(len(self.data)/2)],self.data[int(len(self.data)/2):],self.labels[0:int(len(self.labels)/2)],self.labels[int(len(self.labels)/2):]
			elif v_type == "c_validation":
				train_data,test_data,train_labels,test_labels = cross_validation.train_test_split(self.data, self.labels, test_size=0.3, random_state=0)
			elif v_type =="k_fold":
				train_data,train_labels,test_data,test_labels =  self.s_k_fold(self.data,self.labels,n_folds=n_folds)
			return train_data,train_labels,test_data,test_labels
	def __init__(self,data,labels,split_type="k_fold",n_folds=4):
		self.data = data
		self.classify.data = data
		self.labels = labels
		self.classify.labels = labels
		self.train_data,self.train_labels,self.test_data,self.test_labels = self.split_data(split_type,n_folds=n_folds)
		self.classify.train_data,self.classify.train_labels,self.classify.test_data,self.classify.test_labels = self.train_data,self.train_labels,self.test_data,self.test_labels
	class classify:
		results = []
		default_classifiers = [GaussianNB(),MultinomialNB(),SVC(kernel="linear", C=0.025),SVC(gamma=2, C=1),KNeighborsClassifier(n_neighbors=4),DecisionTreeClassifier(),AdaBoostClassifier(),RandomForestClassifier(),QDA(),LDA(),LogisticRegression(),GradientBoostingClassifier(), ExtraTreesClassifier(),RidgeClassifier(),PassiveAggressiveClassifier(),SGDClassifier()]
		def __init__(self,target_names=None,fun=default_classifiers):
			target_names = self.rock_names
			if not(type(fun) is list):
				fun = [fun]
			for i in fun:
				clf = i
				clf.fit(self.train_data, self.train_labels)
				y_pred = clf.predict(self.test_data)
				try:
					y_confidence = clf.predict_proba(self.train_data)
				except:
					y_confidence = []
				pure_accuracy_rate = len([y_pred[x] for x in range(len(y_pred)) if y_pred[x] == self.test_labels[x]])/float(len(self.test_labels))
				report = classification_report(y_pred, self.test_labels, target_names=target_names)
				cm = confusion_matrix(self.test_labels, y_pred)
				custom_report = self.precision_breakdown(self.test_labels,y_pred,target_names)
				custom_report["overall_accuracy"] = pure_accuracy_rate
				self.results.append({"accuracy":pure_accuracy_rate,"report":report,"custom_report":custom_report,"y_pred":y_pred,"test_labels":self.test_labels,"test_data":self.test_data,"classifier":clf,"confusion_matrix":cm,"y_confidence":y_confidence,"classifier_name":i.__class__.__name__})
		
		def save_classifiers(self,dir="./"):
			for i in self.results:
				joblib.dump(i["classifier"],"{}/{}.pkl".format(dir,i["classifier_name"]), compress=9)
		@staticmethod
		def precision_breakdown(labels,y_pred,target_names):
				precisions = {}
				for i in range(len(target_names)):
					target_equalities = [[labels[x],y_pred[x]] for x in range(len(labels)) if labels[x] == i]
					hits = [x for x in target_equalities if x[0] == x[1]]
					precisions[target_names[i]] = len(hits)/float(len(target_equalities))
				return precisions
		@staticmethod
		def exportDecisionTree(clf,filename):
			out = StringIO()
			out = tree.export_graphviz(clf, out_file=out)
			graph_from_dot_data(out.getvalue()).write_pdf(filename)
			return out
		def confusion_matricies(self,plot=False,save=True,accuracy=None,directory="./"):
			def normalize_mat(cm):
				cm = cm.tolist()
				for i in range(len(cm)):
					rsum = np.sum(cm[i])
					for x in range(len(cm[0])):
						cm[i][x] = float(cm[i][x])/rsum
				return np.array(cm)
			results = self.get_results(sort=True,reverse=True)
			for i in range(len(results)):
				fig = plt.figure()
				ax0 = fig.add_subplot(1,1,1)				
				im = ax0.imshow(normalize_mat(results[i]["confusion_matrix"]), interpolation='nearest',cmap='Reds')
				plt.xticks(np.arange(0,4), self.rock_names)
				plt.yticks(np.arange(0,4), self.rock_names)
				cbar = plt.colorbar(im)
				ax0.set_title("{} Confusion Matrix\n {} accuracy".format(results[i]["classifier_name"],results[i]["accuracy"]))
				if save == True:
					plt.savefig("{}/{}-{}_confusion_matrix.png".format(directory,i+1,results[i]["classifier_name"])) 
			if plot == True:
				plt.show()
			return self
		def plotAnalysis(self,cmap = {1:"red",2:"green",3:"blue",4:"orange"},plot_vars = ["SIO2","AL2O3","MGO"],plot=False,save=True,directory="./",pca=False):
			test_data = self.test_data
			test_labels = self.test_labels
			for i in self.results:
				y_pred = i["y_pred"]
				analysis_type = i["classifier_name"]
				accuracy = i["accuracy"]
				fig = plt.figure()
				ax = fig.add_subplot(111, projection='3d')
				x1,y1,z1 = [test_data[:,self.chemicals.tolist().index(x)] for x in plot_vars]
				hit_shapes = {True:"*",False:"o"}

				hit_mat = [y_pred[i] == test_labels[i] for i in range(len(test_labels))]
				hit_shapes_mat = [hit_shapes[i] for i in hit_mat]
				rock_colors = [cmap[x+1] for x in test_labels]
				for i in range(len(hit_shapes_mat)):
					ax.scatter(x1[i], y1[i], zs=z1[i], c=rock_colors[i], marker=hit_shapes_mat[i],s=50)
				proxy_labels = [matplotlib.lines.Line2D([0],[0], linestyle="none", c='black', marker = 'o'), matplotlib.lines.Line2D([0],[0], linestyle="none", c='black', marker = '*')]
				for i in range(1,5):
					proxy_labels.append(matplotlib.lines.Line2D([0],[0], linestyle="none", c=cmap[i], marker = 'o'))
				ax.legend(proxy_labels, ['Incorrectly Classified', 'Correctly Classified']+self.rock_names.tolist(), numpoints = 1,prop={'size':6},bbox_to_anchor=(1.1, .1))
				ax.set_xlabel(plot_vars[0])
				ax.set_ylabel(plot_vars[1])
				ax.set_zlabel(plot_vars[2])
				ax.view_init(elev=57,azim=82)
				ax.set_title(analysis_type+": "+":\n Overall Accuracy: "+"%.2f%%" % (accuracy*100))
				if save == True:
					plt.savefig("{}/{}_plot.png".format(directory,analysis_type)) 
			if plot == True:plt.show()
			return self
		def plotAnalysisPCA(self,cmap = {1:"red",2:"green",3:"blue",4:"orange"},plot=False,save=True,directory="./"):
			test_data = self.test_data
			test_labels = self.test_labels
			results = self.get_results(sort=True)
			pca = PCA(n_components=3)
			x = pca.fit(test_data)
			raw_data_reduced = PCA.transform(x,test_data)
			for c in range(len(results)):
				y_pred = results[c]["y_pred"]
				analysis_type = results[c]["classifier_name"]
				accuracy = results[c]["accuracy"]
				fig = plt.figure()
				ax = fig.add_subplot(111, projection='3d')
				x1,y1,z1 = [raw_data_reduced[:,x] for x in range(len(raw_data_reduced[0]))]
				r_samples = np.array([[np.array([x1[i],y1[i],z1[i],test_labels[i],y_pred[i]]) for i in range(len(test_labels)) if test_labels[i] == c] for c in range(0,4)]) 
				hit_shapes = {True:"*",False:"o"}
				hit_mat = [y_pred[i] == test_labels[i] for i in range(len(test_labels))]
				hit_shapes_mat = [hit_shapes[i] for i in hit_mat]
				for x in range(len(r_samples)):
					samples = [[i for i in r_samples[x] if i[3] == i[4]],[i for i in r_samples[x] if i[3] != i[4]]]
					try:
						ax.scatter(np.array(samples[0])[:,0],np.array(samples[0])[:,1],zs=np.array(samples[0])[:,2],c=cmap[x+1],marker = '*')
						ax.scatter(np.array(samples[1])[:,0],np.array(samples[1])[:,1],zs=np.array(samples[1])[:,2],c=cmap[x+1],marker = 'o')
					except Exception,err:
						print len(samples[1])
						print analysis_type
						print err
				proxy_labels = [matplotlib.lines.Line2D([0],[0], linestyle="none", c='black', marker = 'o'), matplotlib.lines.Line2D([0],[0], linestyle="none", c='black', marker = '*')]
				for i in range(1,5):
					proxy_labels.append(matplotlib.lines.Line2D([0],[0], linestyle="none", c=cmap[i], marker = 'o'))
				ax.legend(proxy_labels, ['Incorrectly Classified', 'Correctly Classified']+self.rock_names.tolist(), numpoints = 1,prop={'size':6},bbox_to_anchor=(1.1, .1))
				ax.set_xlabel("PCA Component 1")
				ax.set_ylabel("PCA Component 2")
				ax.set_zlabel("PCA Component 3")
				ax.set_title(analysis_type+": "+":\n Overall Accuracy: "+"%.2f%%" % (accuracy*100))
				if save == True:
					plt.savefig("{}/{}-{}_plot.png".format(directory,c,analysis_type)) 
			if plot == True:plt.show()
			return self
		def bar_chart(self,title="Classifier Comparison",horizontal=True,show=False,filename="barchart.png"):
			cmap = {1:"yellow",2:"green",3:"blue",4:"orange",5:"red"}
			hatch = {1:None,2:None,3:None,4:None,5:"\\"}
			fig = plt.figure(figsize=(24.0, 10.0)) if not(horizontal) else plt.figure()
			ax = fig.add_subplot(1,1,1)
			res = np.array(self.get_results("classifier_name","custom_report",sort=True))
			vals = res[:,1].tolist()
			keys = vals[0].keys()
			width = .15
			rects = []
			breakdown = [[i[key] for i in vals] for key in keys[:5]]
			if horizontal:
				names = [x[:] for x in res[:,0].tolist()]
				N = len(names)
				ind = np.arange(N) 
				for i in range(len(breakdown)):
					rects.append(ax.barh(ind+width*(i), breakdown[i], width, color=cmap[i+1],hatch=hatch[i+1]))
				ax.set_ylabel('Classifier Name')
				ax.set_xlabel("Accuracy")
				# plt.subplots_adjust(bottom=0.1, right=0.5, top=0.9)
				ax.set_yticks(ind+width*2.5)
				ax.set_yticklabels(names)
			else:
				names = [x[:12] for x in res[:,0].tolist()]
				N = len(names)
				ind = np.arange(N) 
				for i in range(len(breakdown)):
					rects.append(ax.bar(ind+width*(i), breakdown[i], width, color=cmap[i+1],hatch=hatch[i+1]))
				ax.set_ylabel('Accuracy')
				ax.set_xlabel("Classifier Name")
				ax.set_xticks(ind+width*2.5)
				ax.set_xticklabels(names)
			ax.legend( (i[0] for i in rects), map(lambda x: x.title().replace("_"," "),keys))
			ax.set_title(title)
			ax.tick_params(axis='both', which='major', labelsize=8,direction='out')
			if show:
				plt.show()
			else:
				plt.savefig(filename)
			return self

		def oversized_legend(self,filename, cmap={1:"red",2:"green",3:"blue",4:"orange"}):
			proxy_labels = [matplotlib.lines.Line2D([0],[0], linestyle="none", c='black', marker = 'o'), matplotlib.lines.Line2D([0],[0], linestyle="none", c='black', marker = '*')]
			for i in range(1,5):
				proxy_labels.append(matplotlib.lines.Line2D([0],[0], linestyle="none", c=cmap[i], marker = 'o'))
			fig = pylab.figure()
			figlegend = pylab.figure(figsize=(3,2))
			ax = fig.add_subplot(111)
			figlegend.legend(proxy_labels,['Incorrectly Classified', 'Correctly Classified']+self.rock_names.tolist(),numpoints=1)
			figlegend.savefig(filename)
		def get_avg_performance(self,key="accuracy"):
			su = []
			for i in self.results:
				su.append(i[key])
			try:
				return float(np.sum(su))/len(self.results)
			except:
				return su
		def save_results(self,filename,sort=False,reverse=True):
			sorted_results = sorted(self.results,key=lambda x:x["accuracy"],reverse=reverse)
			handle = open("out__{}".format(filename),"w")
			results = sorted_results if sort else self.results
			for i in results:
				handle.write("{}\n{}\n{}{}".format(i["classifier_name"],i["accuracy"],i["report"],"\n"*5))
			return self
		def get_results(self,*args,**kwargs):
			arglen = len(list(args))
			p = kwargs.get("p",False)
			sort = kwargs.get("sort",False)
			reverse = kwargs.get("reverse",True)
			results = sorted(self.results,key=lambda x:x["accuracy"],reverse=reverse) if sort else self.results
			if p:
				for i in results:
					if arglen > 0:
						for i in [[i[x] for x in args] for i in results] if all([key in results[0] for key in args]) else ["Invalid Key"]: 
							for x in i: print x,"\n"
					else:	
						print "{}\n{}\n{}{}".format(i["classifier_name"],i["accuracy"],i["report"],"\n"*5)
			return np.array(results) if arglen<1 else np.array([[i[x] for x in args] for i in results]) if all([key in results[0] for key in args]) else "Invalid Key"







# RockClassifier.feed("./data/STAMCM.csv").classify().bar_chart(show=True,horizontal=False)





