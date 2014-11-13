import pandas as pd


rock_types={'DUNITE' :1,'HARZBURGITE':2, 'LHERZOLITE':3, 'WEHRLITE':4}
comparison = pd.read_csv("peridotites_clean_SCAM.csv")
prediction = pd.read_csv("peridotites_clean_SCAM copy_classified.csv")

def list_to_rocks(labels):
	return [num_to_rock[x] if x<5 else "error" for x in labels]




comparison = comparison[comparison.columns[0]].tolist()
prediction = prediction[prediction.columns[-1]].tolist()

print "row number, true classification, classifier's classification"
for i in range(len(comparison)):
	print  str(i) + ": " + comparison[i] + ", " + prediction[i]


for i in rock_types.keys():
	print "{} correct: {} pecent".format(i,len([x for x in range(len(comparison)) if (comparison[x] == i) & (comparison[x] == prediction[x])])/float(
		len([x for x in range(len(comparison)) if comparison[x] == i]))*100)