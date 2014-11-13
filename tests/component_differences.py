import pandas as pd 
import numpy as np 
import math 
import matplotlib.pyplot as plt 
from scipy.interpolate import spline
 
import scipy.stats as st
from scipy.optimize import curve_fit
perms = pd.read_csv("./data/perms_test.csv")


vals = perms.values[:,2:]
# len([x for x in vals[i] if type(i) is str])])

# for i in vals:print i

results = []
results2 = []

# print type(str(np.nan))
curr = 3
for i in vals:
	# items = [x for x in i if (type(x) is str)]
	items = [x for x in i if (str(x) != "nan")]
	if len(items[:-1]) == curr:
		curr += 1
		results.append([len(items)-1,float(items[-1])])
for i in vals:
	items = [x for x in i if (str(x) != "nan")]
	results2.append([len(items)-1,float(items[-1])])
# print results,"\n"*5,results2


fig = plt.figure()
ax = fig.add_subplot(1,1,1)
results = np.array(results)

T = results[:,0]
power = results[:,1]
xnew = np.linspace(T.min(),T.max(),len(power))




# ax.plot(xnew,y_fit)
results,results2 = np.array(results),np.array(results2)

ax.plot(results[:,0],results[:,1],c="red")
ax.plot(results2[:,0],results2[:,1],c="blue")
# plt.show()
ax.set_title("Comparison of the Mean Accuracy of All Classifiers Using Different Numbers of Oxides",fontsize=13,y=1.04,horizontalalignment="center")
ax.set_xlabel("Number of Components")
ax.set_ylabel("Classifier Accuracy")

px = np.linspace(0,4,50)


def fitFunc(t, a, b, c):
    return a*np.exp(-b*t) + c
# print power
# print T
print power
print np.exp(power)

# plt.savefig("./n_component_analysis.png")










