import utils
import etl
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_svmlight_file
#Note: You can reuse code that you wrote in etl.py and models.py and cross.py over here. It might help.
# PLEASE USE THE GIVEN FUNCTION NAME, DO NOT CHANGE IT

'''
You may generate your own features over here.
Note that for the test data, all events are already filtered such that they fall in the observation window of their respective patients. Thus, if you were to generate features similar to those you constructed in code/etl.py for the test data, all you have to do is aggregate events for each patient.
IMPORTANT: Store your test data features in a file called "test_features.txt" where each line has the
patient_id followed by a space and the corresponding feature in sparse format.
Eg of a line:
60 971:1.000000 988:1.000000 1648:1.000000 1717:1.000000 2798:0.364078 3005:0.367953 3049:0.013514
Here, 60 is the patient id and 971:1.000000 988:1.000000 1648:1.000000 1717:1.000000 2798:0.364078 3005:0.367953 3049:0.013514 is the feature for the patient with id 60.

Save the file as "test_features.txt" and save it inside the folder deliverables

input:
output: X_train,Y_train,X_test
'''
def my_features():
	#TODO: complete this

	# The features' dimensionality will be reduced in the method below. I chose to do it there as it makes the code cleaner.
	X_train, Y_train = utils.get_data_from_svmlight("../deliverables/features_svmlight.train")
	
    # Test features
	events = pd.read_csv('../data/test/events.csv')
	feature_map = pd.read_csv('../data/test/event_feature_map.csv')

	aggregates_events = etl.aggregate_events(events, None, feature_map, '../deliverables/')

	from collections import defaultdict

	patient_features = defaultdict(list)
	for _, row in aggregates_events.iterrows():
		patient_features[row['patient_id']].append( (row['feature_id'], row['feature_value']) )

	del1 = ''
	del2 = ''

	for key, value in sorted(patient_features.items()):
		del1 += str(int(key)) + ' '
		del2 += str(1) + ' '
		value = sorted(value)
		for v in value:
			del1 += str(int(v[0])) + ':' + str(format(v[1], '.6f')) + ' '
			del2 += str(int(v[0])) + ':' + str(format(v[1], '.6f')) + ' '
		del1 += '\n'
		del2 += '\n'
        
	deliverable2 = open('../deliverables/test_features.txt', 'wb')
	deliverable2.write(bytes((del1),'UTF-8'))
	deliverable2.close()

	deliverable1 = open('../deliverables/test_mfeatures.train', 'wb')
	deliverable1.write(bytes((del2),'UTF-8'))
	deliverable1.close()

	data = load_svmlight_file('../deliverables/test_mfeatures.train', n_features=3190)
	X_test = data[0]

	return X_train, Y_train, X_test


'''
You can use any model you wish.

input: X_train, Y_train, X_test
output: Y_pred
'''
def my_classifier_predictions(X_train,Y_train,X_test):
	#TODO: complete this

	# These parameters were found to be the best via RandomizedSearchCV
	pipe = Pipeline(steps=[
		('svd', TruncatedSVD(n_components=175)),
		('lr', LogisticRegression(C=1, max_iter=200, penalty='l2', solver='liblinear', tol=1e-6))
	])

	pipe.fit(X_train, Y_train)

	Y_pred = pipe.predict(X_test)

	return Y_pred


def main():
	X_train, Y_train, X_test = my_features()
	Y_pred = my_classifier_predictions(X_train,Y_train,X_test)
	utils.generate_submission("../deliverables/test_features.txt",Y_pred)
	#The above function will generate a csv file of (patient_id,predicted label) and will be saved as "my_predictions.csv" in the deliverables folder.

if __name__ == "__main__":
    main()

	