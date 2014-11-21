from multiclassifier import RockClassifier
from sklearn.tree import DecisionTreeClassifier

def main():
	RockClassifier.feed("./data/STAMCM.csv").classify().save_classifiers("./classifiers/")

if __name__ == "__main__":
	main()