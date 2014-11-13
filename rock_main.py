__author__ = 'Michael  grossberg'

import csv
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import mlab
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cross_validation import StratifiedKFold
from scipy.cluster.vq import whiten as simple_whiten
#from scikits.learn.lda import LDA
from sklearn.lda import LDA
#from scikits.learn.qda import QDA
from sklearn.qda import QDA
#from scikits.learn.metrics import classification_report
from sklearn.metrics import classification_report
#from scikits.learn.metrics import confusion_matrix
#from scikits.learn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from mayavi import mlab as mv_mlab
from matplotlib.colors import ColorConverter
cc = ColorConverter()

label_field = 'ROCK NAME'
def main():
    filenames = ["peridotites_clean_complete.csv",
                 "peridotites_clean_SCAM.csv",
                 "BakerBeckettAsimowData.csv"]
    filename =filenames[0]
    print("Dataset: "+filename)
    labels, feature_mat,feature_fields = read_data(filename)
    unique_labels = get_unique_labels(labels)
    ylabels = np.array([unique_labels.index(label) for label in labels])
    n_classes = len(unique_labels)
    n_samples, n_features = feature_mat.shape
    print "Total dataset size:"
    print "n_samples: %d" % n_samples
    print "n_features: %d" % n_features
    print "n_classes: %d" % n_classes

    # proj = fisher_analysis(feature_mat,labels,unique_labels)

    # Split into a training set and a test set using a stratified k fold

    # split into a training and testing set
    train, test = iter(StratifiedKFold(labels, k=4)).next()
    print "n_training_examples: %d" % len(train)
    print "n_testing_examples: %d" % len(test)
    feature_train, feature_test = feature_mat[train], feature_mat[test]
    ylabels_train, ylabels_test = ylabels[train], ylabels[test]
    X,y = feature_train,ylabels_train
    X_test,y_test = feature_test,ylabels_test

    print y 
    print  ylabels_train, ylabels_test


#     print("Linear Discriminant Analysis")
#     clf = LDA()
#     clf.fit(X,y)
#     y_pred =  clf.predict(X_test)
#     print classification_report(y_test,y_pred, target_names=unique_labels)
#     #print confusion_matrix(y_test, y_pred, labels=range(n_classes))
#     print("Quadradic Discriminant Analysis")
#     clf = QDA()
#     clf.fit(X,y)
#     y_pred =  clf.predict(X_test)
#     print classification_report(y_test,y_pred, target_names=unique_labels)
#     print("Nearest Neighbor Classifier")
#     neigh = NeighborsClassifier(n_neighbors=1)
#     neigh.fit(X,y)
#     y_pred = neigh.predict(X_test)
#     print classification_report(y_test,y_pred, target_names=unique_labels)
#     #pca_analysis(feature_mat,labels,unique_labels)
#     print("Fisher Space 3d Classifier")
#     proj3 = proj[:3,:]
#     print "proj3.shape = ",proj3.shape
#     print "feature_train.shape = ",feature_train.shape
#     X,y = np.dot(proj3,feature_train.T).T,ylabels_train
#     X_test,y_test = np.dot(proj3,feature_test.T).T,ylabels_test
#     clf = LDA()
#     clf.fit(X,y)
#     y_pred =  clf.predict(X_test)
#     print classification_report(y_test,y_pred, target_names=unique_labels)
#     fig3 = plt.figure()
#     ax = fig3.add_subplot(1,1,1)
#     xinds = np.arange(0,len(feature_fields))
#     width = 0.35
#     ax.bar(xinds+(.5-(width/2.)),proj3[0,:],width=width)
#     ax.set_xticks(xinds+.5)
#     ax.set_xticklabels(feature_fields)
#     for xind in xinds+.5:
#         ax.plot([xind,xind],[-0.5,0.5],'k:')
#     ax.set_title("First Fisher Component")
#     fig3.savefig("first_fisher_component.png")
#     fig4 = plt.figure()
#     ax = fig4.add_subplot(1,1,1)
#     xinds = np.arange(0,len(feature_fields))
#     width = 0.35
#     ax.bar(xinds+(.5-(width/2.)),proj3[1,:],width=width)
#     ax.set_xticks(xinds+.5)
#     ax.set_xticklabels(feature_fields)
#     for xind in xinds+.5:
#         ax.plot([xind,xind],[-0.5,0.5],'k:')
#     ax.set_title("Second Fisher Component")
#     fig4.savefig("second_fisher_component.png")
#     fig5 = plt.figure()
#     ax = fig5.add_subplot(1,1,1)
#     xinds = np.arange(0,len(feature_fields))
#     width = 0.35
#     ax.bar(xinds+(.5-(width/2.)),proj3[2,:],width=width)
#     ax.set_xticks(xinds+.5)
#     ax.set_xticklabels(feature_fields)
#     for xind in xinds+.5:
#         ax.plot([xind,xind],[-0.5,0.5],'k:')
#     ax.set_title("Third Fisher Component")
#     fig5.savefig("third_fisher_component.png")
#     plt.show()

# def fisher_analysis(feature_mat,labels,unique_labels):
#     #wfeature_mat = simple_whiten(feature_mat)
#     data_mu = feature_mat.mean(axis=0)
#     scatter_shape = feature_mat.shape[1]
#     seg_data = []
#     class_means = []
#     scatter_ws = []
#     for rocktype in unique_labels:
#         inds = np.array([rocktype==label for label in labels]).nonzero()
#         class_mat = np.squeeze(feature_mat[inds,:])
#         seg_data.append(class_mat)
#         class_mean = class_mat.mean(axis=0)
#         scatter_w_c = class_mat - class_mean
#         scatter_w_c = np.dot(scatter_w_c.T,scatter_w_c)
#         scatter_ws.append(scatter_w_c.ravel())
#         class_means.append(class_mean)

#     scatter_w = np.vstack(scatter_ws).mean(axis=0).reshape((scatter_shape,scatter_shape))
#     scatter_b = np.array(class_means)-data_mu
#     scatter_b = np.dot(scatter_b.T,scatter_b)
#     pca = mlab.PCA(np.dot(np.linalg.pinv(scatter_w),scatter_b))
#     proj = pca.Wt

#     fig0 = plt.figure()
#     ax= fig0.add_subplot(1,1,1)
#     colors = 'rgbcm'

#     for ind,rocktype in enumerate(unique_labels):
#         projected_data = np.dot(proj[:2,:],seg_data[ind].T).T
#         x= projected_data[:,0]
#         y = projected_data[:,1]
#         ax.scatter(x,y,color=colors[ind],label=rocktype)
#     ax.legend(loc=3)
#     ax.set_title("Fisher Two Components")
#     ax.set_xlabel("First Fisher Component")
#     ax.set_ylabel("Second Fisher Component")
#     fig0.savefig('fisher_two.png')
#     fig1 = plt.figure()

#     ax = fig1.add_subplot(111, projection='3d')
#     projected_data_list=[]
#     for ind,rocktype in enumerate(unique_labels):
#         projected_data = np.dot(proj[:3,:],seg_data[ind].T).T
#         projected_data_list.append(projected_data)

#     for ind,rocktype in enumerate(unique_labels):
#         projected_data = projected_data_list[ind]
#         x = projected_data[:,0]
#         y = projected_data[:,1]
#         z = projected_data[:,2]
#         ax.scatter(x,y,z,color=colors[ind],label=rocktype)
#     ax.set_xlabel('First Fisher Component')
#     ax.set_ylabel('Second Fisher Component')
#     ax.set_zlabel('Third Fisher Component')

#     ax.set_title("Fisher Three Components")
#     fig1.savefig("fisher_three.png")
#     if False:
#         for ind,rocktype in enumerate(unique_labels):
#             projected_data = projected_data_list[ind]
#             x = projected_data[:,0]
#             y = projected_data[:,1]
#             z = projected_data[:,2]

#             color = cc.to_rgb(colors[ind])
#             mv_mlab.points3d(x,y,z,color=color,scale_factor=0.2)
#         mv_mlab.axes()
#         mv_mlab.show()



#     return proj




# def pca_analysis(feature_mat,labels,unique_labels):

#     wfeature_mat = simple_whiten(feature_mat)
#     seg_data = []
#     for rocktype in unique_labels:
#         inds = np.array([rocktype==label for label in labels]).nonzero()
#         seg_data.append(np.squeeze(wfeature_mat[inds,:]))


#     pca = mlab.PCA(wfeature_mat)
#     proj = pca.Wt


#     fig0 = plt.figure()
#     ax= fig0.add_subplot(1,1,1)
#     colors = 'rgbcm'

#     for ind,rocktype in enumerate(unique_labels):
#         projected_data = np.dot(proj[:2,:],seg_data[ind].T).T
#         x= projected_data[:,0]
#         y = projected_data[:,1]
#         ax.scatter(x,y,color=colors[ind],label=rocktype)
#     ax.legend(loc=3)
#     ax.set_title("Whiten Two Principle Components")
#     fig1 = plt.figure()

#     ax = fig1.add_subplot(111, projection='3d')
#     for ind,rocktype in enumerate(unique_labels):
#         projected_data = np.dot(proj[:3,:],seg_data[ind].T).T
#         x = projected_data[:,0]
#         y = projected_data[:,1]
#         z = projected_data[:,2]
#         ax.scatter(x,y,z,color=colors[ind],label=rocktype)

#     ax.set_title("Three Principle Components")

#     #plt.show()
#     return


def read_data(filename):
    labels, feature_mat = None, None
    with open(filename,'rU') as csvfile:
        dialect = csv.Sniffer().sniff(csvfile.read(8024))
        csvfile.seek(0)
        dr = csv.DictReader(csvfile, dialect=dialect)
        data = [row for row in dr]
        labels = np.array([row[label_field] for row in data])
        feature_fields = [field for field in data[0].keys() if field != label_field]
        try:
            feature_mat = np.squeeze(np.array([[float(row[field]) for field in feature_fields] for row in data]))
        except Exception, e:
            print "row[",field,"]=", row[field]
            print "row =", row
            raise e
    return labels, feature_mat, feature_fields
    
def get_unique_labels(labels):
    return list(set(labels))


if __name__ == '__main__':
    main()
  
