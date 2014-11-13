import numpy as np
import pandas as pd
import scipy as st
from rocksep_utils import *
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import matplotlib.pyplot as plt


cmap = {1:"red",2:"green",3:"blue",4:"orange"}
rock_types={'DUNITE' :1,'HARZBURGITE':2, 'LHERZOLITE':3, 'WEHRLITE':4}
inv_map = {v: k for k, v in rock_types.items()}


pca = PCA(n_components=3)
x = pca.fit(raw_data)
raw_data_reduced = PCA.transform(x,raw_data)

x,y,z = [raw_data_reduced[:,x] for x in range(len(raw_data_reduced[0]))]


r_samples = np.array( [ [np.array([x[i],y[i],z[i]]) for i in range(len(nlabels)) if nlabels[i] == c] for c in range(1,5) ] ) 
fig = plt.figure()
ax = fig.add_subplot(111,projection="3d")

for i in range(len(r_samples)):
	ax.scatter(np.array(r_samples[i])[:,0],np.array(r_samples[i])[:,1],zs=np.array(r_samples[i])[:,2],c=cmap[i+1])


proxies = [matplotlib.lines.Line2D([0],[0], linestyle="none", c=cmap[i+1], marker = 'o') for i in range(len(r_samples))]
prox_labels = [inv_map[i+1] for i in range(len(r_samples))];

plt.legend(proxies,prox_labels,numpoints = 1,prop={'size':6},bbox_to_anchor=(1.1, .1))

ax.view_init(elev=57,azim=82)
ax.set_title("Principle Component Analysis of SCAM Values")

ax.set_xlabel('PCA Component 1')
ax.set_ylabel('PCA Component 2')
ax.set_zlabel('PCA Component 3')

plt.savefig("pca_analysis.png")
