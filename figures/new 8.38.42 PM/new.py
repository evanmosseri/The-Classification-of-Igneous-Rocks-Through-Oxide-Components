from sklearn.externals import joblib

x = joblib.load("test.pkl")	

print x.predict([38.8,0.24,0.4,59.21])