# full_data = pd.read_csv("./data/peridotites_clean_complete.csv")
# columns = full_data.columns.tolist()
# perm = columns[1:]

# comb = list(itertools.combinations(perm,4))

# avg = []
# rock_names = np.unique(full_data[full_data.keys()[0]]).tolist()

# for i in comb:
# 	params = ["ROCK NAME"]+list(i)
# 	temp_data = full_data[params].values
# 	np.random.shuffle(temp_data)
# 	label = temp_data[:,0]
# 	dat = temp_data[:,1:]
# 	avg.append([params]+[avg_classifier(RockClassifier(np.array(dat,dtype=np.float64),numbered_labels(label,rock_types)).classify(rock_names,fun=classifiers).get_results())])




# combs = [list(itertools.combinations(perm,i)) for i in range(3,len(perm)+1)]

# for comb in combs:
	# for i in comb:
		# params = ["ROCK NAME"]+list(i)
		# temp_data = full_data[params].values
		# np.random.shuffle(temp_data)
		# label = temp_data[:,0]
		# dat = temp_data[:,1:]
		# avg.append(params+[RockClassifier(np.array(dat,dtype=np.float64),numbered_labels(label,rock_types)).classify(rock_names,fun=classifiers).get_avg_performance()])
		# avg.append(params+[""]*(len(perm)+1-len(params))+[avg_classifier(RockClassifier(np.array(dat,dtype=np.float64),numbered_labels(label,rock_types)).classify(rock_names,fun=classifiers).get_results())])




# print np.mean(np.array(avg)[:,1])

# print avg[np.argmax(np.array(avg)[:,1])]


# pd.DataFrame(avg).to_csv("perms_test.csv")


# m = 0
# ind = 0;

# for i in range(len(avg)):
# 	if avg[i][1] > m:
# 		m = avg[i][1]
# 		ind = i

# print m,ind,avg[ind]


# print avg[np.argmax(np.array(avg)[:,1])]

# rock_names = np.unique(pd_data[pd_data.keys()[0]]).tolist()
# random_data = pd_data.values

# np.random.shuffle(random_data)









