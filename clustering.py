import numpy as np
import pandas as pd
import scipy as st
from rocksep_utils import *
from sklearn.cluster import KMeans


km = KMeans(n_clusters=4,random_state=0)
km.fit(raw_data)


