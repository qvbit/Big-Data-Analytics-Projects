import models_partc
from sklearn.model_selection import KFold, ShuffleSplit
from numpy import mean
import functools
from sklearn.linear_model import LogisticRegression

import utils

# USE THE GIVEN FUNCTION NAME, DO NOT CHANGE IT

# USE THIS RANDOM STATE FOR ALL OF YOUR CROSS VALIDATION TESTS, OR THE TESTS WILL NEVER PASS
RANDOM_STATE = 545510477

#input: training data and corresponding labels
#output: accuracy, auc
def get_acc_auc_kfold(X,Y,k=5):
	#TODO:First get the train indices and test indices for each iteration
	#Then train the classifier accordingly
	#Report the mean accuracy and mean auc of all the folds
	kf = KFold(n_splits=k, random_state=RANDOM_STATE)
	lr = LogisticRegression(random_state=RANDOM_STATE)
	scores = []

	for train_idx, test_idx in kf.split(X):
		X_train, Y_train = X[train_idx], Y[train_idx]
		X_test, Y_test = X[test_idx], Y[test_idx]
		lr.fit(X_train, Y_train)
		Y_pred = lr.predict(X_test)
		acc, auc, _, _, _ = models_partc.classification_metrics(Y_pred, Y_test)
		scores.append((acc, auc))

	scores = list(zip(*scores))
	acc_mean = functools.reduce(lambda x, y: x + y, scores[0]) / len(scores[0])
	auc_mean = functools.reduce(lambda x, y: x + y, scores[1]) / len(scores[1])

	return acc_mean, auc_mean


#input: training data and corresponding labels
#output: accuracy, auc
def get_acc_auc_randomisedCV(X,Y,iterNo=5,test_percent=0.2):
	#TODO: First get the train indices and test indices for each iteration
	#Then train the classifier accordingly
	#Report the mean accuracy and mean auc of all the iterations
	sf = ShuffleSplit(n_splits=5, test_size=test_percent, random_state=RANDOM_STATE)

	lr = LogisticRegression(random_state=RANDOM_STATE)
	scores = []

	for train_idx, test_idx in sf.split(X):
		X_train, Y_train = X[train_idx], Y[train_idx]
		X_test, Y_test = X[test_idx], Y[test_idx]
		lr.fit(X_train, Y_train)
		Y_pred = lr.predict(X_test)
		acc, auc, _, _, _ = models_partc.classification_metrics(Y_pred, Y_test)
		scores.append((acc, auc))

	scores = list(zip(*scores))
	acc_mean = functools.reduce(lambda x, y: x + y, scores[0]) / len(scores[0])
	auc_mean = functools.reduce(lambda x, y: x + y, scores[1]) / len(scores[1])

	return acc_mean, auc_mean

def main():
	X,Y = utils.get_data_from_svmlight("../deliverables/features_svmlight.train")
	print("Classifier: Logistic Regression__________")
	acc_k,auc_k = get_acc_auc_kfold(X,Y)
	print(("Average Accuracy in KFold CV: "+str(acc_k)))
	print(("Average AUC in KFold CV: "+str(auc_k)))
	acc_r,auc_r = get_acc_auc_randomisedCV(X,Y)
	print(("Average Accuracy in Randomised CV: "+str(acc_r)))
	print(("Average AUC in Randomised CV: "+str(auc_r)))

if __name__ == "__main__":
	main()

