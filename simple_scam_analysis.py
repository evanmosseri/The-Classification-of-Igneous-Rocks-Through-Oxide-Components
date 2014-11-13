import numpy as np
import pandas as pd
import scipy as st
from rocksep_utils import *
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import matplotlib.pyplot as plt
import itertools

cmap = {1:"red",2:"green",3:"blue",4:"orange"}
rock_types={'DUNITE' :1,'HARZBURGITE':2, 'LHERZOLITE':3, 'WEHRLITE':4}
inv_map = {v: k for k, v in rock_types.items()}
plot_vars = ["SIO2","AL2O3","MGO"]
scam_perm = [list(x) for x in list(itertools.combinations(scam,3))]



for m in scam_perm:
	x,y,z = [raw_data[:,chemicals.index(x)+1] for x in m]
	alphabet = "ABCD"
	r_samples = np.array( [ [np.array([x[i],y[i],z[i]]) for i in range(len(nlabels)) if nlabels[i] == c] for c in range(1,5) ] ) 
	fig = plt.figure()
	ax = fig.add_subplot(111,projection="3d")

	for i in range(len(r_samples)):
		ax.scatter(np.array(r_samples[i])[:,0],np.array(r_samples[i])[:,1],zs=np.array(r_samples[i])[:,2],c=cmap[i+1])
	# ax.set_title("Simple {}, {}, {} Plot".format(m[0],m[1],m[2]))
	proxy_labels = []
	for i in rock_types.values():
		proxy_labels.append(matplotlib.lines.Line2D([0],[0], linestyle="none", c=cmap[i], marker = 'o'))
	ax.legend(proxy_labels, rock_types.keys(), numpoints = 1,prop={'size':6},bbox_to_anchor=(1.1, .1))
	# ax.set_xlabel(m[0])
	# ax.set_ylabel(m[1])
	# ax.set_zlabel(m[2])
	# print scam_perm
	a,b,c = alphabet[scam.index(m[0])],alphabet[scam.index(m[1])],alphabet[scam.index(m[2])]
	ax.set_title("Simple {}, {}, {} Plot".format(a,b,c))
	ax.set_xlabel(a)
	ax.set_ylabel(b)
	ax.set_zlabel(c)
	ax.view_init(elev=57,azim=82)
	# plt.show()
	plt.savefig("./{}{}{}_simple_plot.png".format(a,b,c))






# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# x1,y1,z1 = [raw_data[:,chemicals.index(x)] for x in plot_vars]
# hit_shapes = {True:"*",False:"o"}

# hit_mat = [y_pred[i] == test_labels[i] for i in range(len(test_labels))]
# hit_shapes_mat = [hit_shapes[i] for i in hit_mat]
# rock_colors = [cmap[x] for x in test_labels]

# for i in range(len(hit_shapes_mat)):
# 	ax.scatter(x1[i], y1[i], zs=z1[i], c=rock_colors[i], marker=hit_shapes_mat[i],s=50)

# proxy_labels = [matplotlib.lines.Line2D([0],[0], linestyle="none", c='black', marker = 'o'), matplotlib.lines.Line2D([0],[0], linestyle="none", c='black', marker = '*')]
# for i in rock_types.values():
# 	proxy_labels.append(matplotlib.lines.Line2D([0],[0], linestyle="none", c=cmap[i], marker = 'o'))
# ax.legend(proxy_labels, ['Incorrectly Classified', 'Correctly Classified']+rock_types.keys(), numpoints = 1,prop={'size':6},bbox_to_anchor=(1.1, .1))
# ax.set_xlabel(plot_vars[0])
# ax.set_ylabel(plot_vars[1])
# ax.set_zlabel(plot_vars[2])
# ax.view_init(elev=57,azim=82)
# ax.set_title(analysis_type+": "+title+":\n Overall Accuracy: "+"%.1f%%" % accuracy)
# if plot == True:
# 	plt.show()
