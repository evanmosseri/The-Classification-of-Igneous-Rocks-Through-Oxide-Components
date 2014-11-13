#!/usr/bin/env python
# encoding: utf-8
"""
untitled.py

Created by Michael Grossberg on 2012-04-12.
Copyright (c) 2012 __MyCompanyName__. All rights reserved.
"""
from mpl_toolkits.mplot3d.axes3d import Axes3D
import csv
import sys
import os
import numpy as np
from sklearn.cross_validation import StratifiedKFold
from sklearn.lda import LDA
from sklearn.qda import QDA
from sklearn.metrics import classification_report
from matplotlib import mlab
from matplotlib import pyplot as plt
from matplotlib import mlab
from scipy.cluster.vq import whiten as simple_whiten
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.externals import joblib

#new files
#dunites_clean_SCAM.txt
#wehrl_matlab_SCAM.txt
#harz_matlab_SCAM.txt
#lherzolite_clean_SCAM.xlsx
#herzolite_clean_SCAM_new.csv
rockfiles = {'dunite':'./data/dunites_clean_SCAM.csv',
              'harzburgite':'./data/harz_matlab_SCAM.csv',
              'wehrlite':'./data/wehrl_matlab_SCAM.csv',
              'lherzolite':'./data/lherzolite_clean_SCAM.csv'
              }
# rockfiles = {'peridotites':'peridotites_clean_complete.csv'}
def create_train_test(feature_mat,ylabels):
    train, test = iter(StratifiedKFold(ylabels, k=4)).next()
    print "n_training_examples: %d" % len(train)
    print "n_testing_examples: %d" % len(test)
    feature_train, feature_test = feature_mat[train], feature_mat[test]
    ylabels_train, ylabels_test = ylabels[train], ylabels[test]
    X,y = feature_train,ylabels_train
    X_test,y_test = feature_test,ylabels_test
    return X,y,X_test,y_test

def LDA_analysis(X,y,X_test,y_test,unique_labels):
    print("Linear Discriminant Analysis")
    clf = LDA()
    clf.fit(X,y)
    y_pred =  clf.predict(X_test)
    pure_accuracy_rate = len([y_pred[x] for x in range(len(y_pred)) if y_pred[x] == y_test[x]])/float(len(y_test))
    print pure_accuracy_rate
    print classification_report(y_test,
                                y_pred, 
                                target_names=unique_labels)
    return y_pred,clf, clf.transform(X_test)

def QDA_analysis(X,y,X_test,y_test,unique_labels):
    print("Quadradic Discriminant Analysis")
    clf = QDA()
    clf.fit(X,y)
    y_pred =  clf.predict(X_test)
    pure_accuracy_rate = len([y_pred[x] for x in range(len(y_pred)) if y_pred[x] == y_test[x]])/float(len(y_test))
    print pure_accuracy_rate
    print classification_report(y_test,
                                y_pred, 
                                target_names=unique_labels)
    return y_pred,clf
    
def DT_analysis(X,y,X_test,y_test,unique_labels):
    print("Decision Tree Analysis")
    clf = tree.DecisionTreeClassifier()
    clf.fit(X,y)
    y_pred =  clf.predict(X_test)
    pure_accuracy_rate = len([y_pred[x] for x in range(len(y_pred)) if y_pred[x] == y_test[x]])/float(len(y_test))
    print pure_accuracy_rate
    print classification_report(y_test,
                                y_pred, 
                                target_names=unique_labels)
    return y_pred,clf

def NN_analysis(X,y,X_test,y_test,unique_labels):
    print("Nearest Neighbors Analysis")
    clf = KNeighborsClassifier(n_neighbors=4)
    clf.fit(X,y)
    y_pred =  clf.predict(X_test)
    pure_accuracy_rate = len([y_pred[x] for x in range(len(y_pred)) if y_pred[x] == y_test[x]])/float(len(y_test))
    print pure_accuracy_rate
    print classification_report(y_test,
                                y_pred, 
                                target_names=unique_labels)
    return y_pred,clf

def RF_analysis(X,y,X_test,y_test,unique_labels):
    print("Random Forest Analysis")
    clf = RandomForestClassifier(n_estimators=4)
    clf.fit(X,y)
    y_pred =  clf.predict(X_test)
    pure_accuracy_rate = len([y_pred[x] for x in range(len(y_pred)) if y_pred[x] == y_test[x]])/float(len(y_test))
    print pure_accuracy_rate
    print classification_report(y_test,
                                y_pred, 
                                target_names=unique_labels)
    return y_pred,clf

def NB_analysis(X,y,X_test,y_test,unique_labels):
    print("Naive Base Analysis")
    clf = GaussianNB()
    clf.fit(X,y)
    y_pred =  clf.predict(X_test)
    pure_accuracy_rate = len([y_pred[x] for x in range(len(y_pred)) if y_pred[x] == y_test[x]])/float(len(y_test))
    print pure_accuracy_rate
    print classification_report(y_test,
                                y_pred, 
                                target_names=unique_labels)
    return y_pred,clf
    
def read_data():
  data ={}
  for rock, filename in rockfiles.items():
      with open(filename,'rU') as csvfile:
          fieldnames =['id','S','C','A','M']
          dialect = csv.Sniffer().sniff(csvfile.read(8024))
          csvfile.seek(0)
          dr = csv.DictReader(csvfile, 
                              dialect=dialect,
                              fieldnames=fieldnames)
          rock_data = []
          for row in dr:
              try:
                  scam_data = (float(row['S']),
                      float(row['C']),
                      float(row['A']),
                      float(row['M']),)
                  rock_data.append(scam_data)
              except ValueError, ex:
                  continue
                  
          data[rock] = np.array(rock_data)
  return data

def process_data(raw_data):
    ylabels = []
    label_key = []
    feature_mat = None
    for ind, label_name in enumerate(raw_data.keys()):
        rows = raw_data[label_name]
        label_key.append(label_name)
        ylabels += [ind]*len(rows)
        if feature_mat == None:
            feature_mat = np.array(rows)
        else:
            feature_mat = np.vstack([feature_mat,np.array(rows)])
    return label_key, np.array(ylabels),feature_mat
    
def pca_analysis(feature_mat,labels,unique_labels):

    wfeature_mat = simple_whiten(feature_mat)
    seg_data = []
    for rock_ind, rocktype in enumerate(unique_labels):
        inds = np.array([rock_ind == label for label in labels]).nonzero()
        seg_data.append(np.squeeze(wfeature_mat[inds,:]))

    pca = mlab.PCA(wfeature_mat)
    proj = pca.Wt
    fig0 = plt.figure()
    ax= fig0.add_subplot(1,1,1)
    colors = 'rgbcm'
    
    for ind,rocktype in enumerate(unique_labels):
        projected_data = np.dot(proj[:2,:],seg_data[ind].T).T

        x = projected_data[:,0]
        y = projected_data[:,1]
        # ax.scatter(x,y,color=colors[ind],label=rocktype)
        ax.scatter(x,y,color='r',label=rocktype)
    ax.legend(loc=3)
    ax.set_title("Whiten Two Principle Components")
    fig1 = plt.figure()

    ax = fig1.add_subplot(111, projection='3d')
    for ind,rocktype in enumerate(unique_labels):
        projected_data = np.dot(proj[:3,:],seg_data[ind].T).T
        x = projected_data[:,0]
        y = projected_data[:,1]
        z = projected_data[:,2]
        ax.scatter(x,y,z,color=colors[ind],label=rocktype)

    ax.set_title("Three Principle Components")

    plt.show()
    return 

     
def main():
    raw_data=read_data()
    label_key, ylabels,feature_mat = process_data(raw_data)

    X,y,X_test,y_test = create_train_test(feature_mat,ylabels)


    classifiers = [
    QDA_analysis(X,y,X_test,y_test,label_key),
    DT_analysis(X,y,X_test,y_test,label_key),
    NN_analysis(X,y,X_test,y_test,label_key),
    RF_analysis(X,y,X_test,y_test,label_key),
    NB_analysis(X,y,X_test,y_test,label_key)    
    ]

    for i in classifiers:
        print i[1]
    # print classifiers[0][0]
        # joblib.dump(i[1], "decision_trees/{}.pkl".format(i[1].__class__.__name__), compress=9)
    # print classifiers[0]

    # y_pred, X_proj = LDA_analysis(X,y,X_test,y_test,label_key)

    # fig0 = plt.figure()
    # ax= fig0.add_subplot(1,1,1)
    # colors = 'rgbcm'
    # for ind,rocktype in enumerate(label_key):
    #     # True positives
    #     class_inds = (np.logical_and(y_test == ind,
    #                   y_pred == ind)).nonzero()
    #     if len(np.squeeze(class_inds)) > 0:
    #         x = X_proj[class_inds,0]
    #         y = X_proj[class_inds,1]
    #         # ax.scatter(x, y, marker='o', color=colors[ind],label='Correctly labeled '+rocktype)
    #     # Misclassified
    #     class_inds = (np.logical_and(y_test == ind,
    #                   y_pred != ind)).nonzero()
    #     if len(np.squeeze(class_inds)) > 0:
    #         #print 'class inds =', class_inds
    #         x = X_proj[class_inds,0]
    #         y = X_proj[class_inds,1]
            # ax.scatter(x, y, marker='x', color=colors[ind],label='Misclassified '+rocktype)
        
    # ax.legend(loc='lower left')
    # ax.set_title("Proj Components")


    
    

    #pca_analysis(feature_mat,ylabels,label_key)
    # plt.show()
    #print len(labels)
    #print feature_mat.shape

if __name__ == '__main__':
    main()

