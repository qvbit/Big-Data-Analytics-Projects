import numpy as np
import pandas as pd
from scipy import sparse
import torch
from torch.utils.data import TensorDataset, Dataset
from sklearn.preprocessing import StandardScaler


def load_seizure_dataset(path, model_type):
	"""
	:param path: a path to the seizure data CSV file
	:return dataset: a TensorDataset consists of a data Tensor and a target Tensor
	"""
	# TODO: Read a csv file from path.
	# TODO: Please refer to the header of the file to locate X and y.
	# TODO: y in the raw data is ranging from 1 to 5. Change it to be from 0 to 4.
	# TODO: Remove the header of CSV file of course.
	# TODO: Do Not change the order of rows.
	# TODO: You can use Pandas if you want to.

	df = pd.read_csv(path)

	if model_type == 'MLP':
		raw = df.loc[:, "X1":"X178"].values
		scaler = StandardScaler()
		scaler.fit(raw)
		data = scaler.transform(raw)
		target = df['y'].values - 1
		dataset = TensorDataset(torch.from_numpy(data.astype('float32')), torch.from_numpy(target.astype('long')))
	elif model_type == 'CNN':
		raw = df.loc[:, "X1":"X178"].values
		scaler = StandardScaler()
		scaler.fit(raw)
		data = scaler.transform(raw)
		target = df['y'].values - 1
		dataset = TensorDataset(torch.from_numpy(data.astype('float32')).unsqueeze(1), torch.from_numpy(target.astype('long')))
	elif model_type == 'RNN':
		data = df.loc[:, "X1":"X178"].values
		target = df['y'].values - 1
		dataset = TensorDataset(torch.from_numpy(data.astype('float32')).unsqueeze(2), torch.from_numpy(target.astype('long')))
	else:
		raise AssertionError("Wrong Model Type!")

	return dataset


def calculate_num_features(seqs):
    """
    :param seqs:
    :return: the calculated number of features
    """
    # TODO: Calculate the number of features (diagnoses codes in the train set)
    import collections

    def flatten(l):
        for el in l:
            if isinstance(el, collections.Iterable):
                for sub in flatten(el):
                    yield sub
            else:
                yield el
                
    return len(set(flatten(seqs)))


class VisitSequenceWithLabelDataset(Dataset):
    def __init__(self, seqs, labels, num_features):
        """
        Args:
            seqs (list): list of patients (list) of visits (list) of codes (int) that contains visit sequences
            labels (list): list of labels (int)
            num_features (int): number of total features available
        """

        if len(seqs) != len(labels):
            raise ValueError("Seqs and Labels have different lengths")

        self.labels = labels

        # TODO: Complete this constructor to make self.seqs as a List of which each element represent visits of a patient
        # TODO: by Numpy matrix where i-th row represents i-th visit and j-th column represent the feature ID j.
        # TODO: You can use Sparse matrix type for memory efficiency if you want.
        
        import numpy as np
        self.num_features = num_features
        n = num_features
        res = []

        for visits in seqs:
            m = len(visits)
            patient_matrix = np.zeros((m , n))
            for i, visit in enumerate(visits):
                for code in visit:
                    patient_matrix[i, int(code)] = 1
            res.append(patient_matrix)
            
        self.seqs = res

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        # returns will be wrapped as List of Tensor(s) by DataLoader
        return self.seqs[index], self.labels[index]


def visit_collate_fn(batch):
    """
    DataLoaderIter call - self.collate_fn([self.dataset[i] for i in indices])
    Thus, 'batch' is a list [(seq1, label1), (seq2, label2), ... , (seqN, labelN)]
    where N is minibatch size, seq is a (Sparse)FloatTensor, and label is a LongTensor

    :returns
        seqs (FloatTensor) - 3D of batch_size X max_length X num_features
        lengths (LongTensor) - 1D of batch_size
        labels (LongTensor) - 1D of batch_size
    """

    # TODO: Return the following two things
    # TODO: 1. a tuple of (Tensor contains the sequence data , Tensor contains the length of each sequence),
    # TODO: 2. Tensor contains the label of each sequence
    
    batch_size = len(batch)
    num_features = len(batch[0][0][0])
    max_length = 0

    labels = []
    lengths = []
    seqs = []

    for i, (seq, label) in enumerate(batch):
        if len(seq) > max_length:
            max_length = len(seq)
        seqs.append(seq)
        labels.append(label)
        lengths.append(len(seq))

    combined = list(zip(seqs, lengths, labels))
    combined.sort(key=lambda x: x[1], reverse=True)

    seqs, lengths, labels = list(zip(*combined))

    seqs_np = np.zeros((batch_size, max_length, num_features))

    for i, seq in enumerate(seqs):
        for j, row in enumerate(seq):
            seqs_np[i, :, :][j] = row
        
    seqs_tensor = torch.FloatTensor(seqs_np)
    lengths_tensor = torch.LongTensor(lengths)
    labels_tensor = torch.LongTensor(labels)

    return (seqs_tensor, lengths_tensor), labels_tensor
