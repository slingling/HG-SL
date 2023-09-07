
import numpy as np

from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, roc_auc_score, average_precision_score


"""
	Utility functions for evaluating the model performance
"""

class Metrics(object):

	def __init__(self):
		super().__init__()

	def compute_metric(self, y_prob, y_true):
		k_list = ['Acc', 'F1', 'Pre', 'Recall']
		y_pre = np.array(y_prob).argmax(axis=1)
		size = len(y_prob)
		assert len(y_prob) == len(y_true)

		scores = {str(k): 0.0 for k in k_list}
		scores['Acc'] += accuracy_score(y_true, y_pre) * size
		scores['F1'] += f1_score(y_true, y_pre, average='macro') * size
		scores['Pre'] += precision_score(y_true, y_pre, zero_division=0) * size
		scores['Recall'] += recall_score(y_true, y_pre, zero_division=0) * size

		# y_true = np.array(y_true)
		# prob_log = y_prob[:, 1].tolist()
		#scores['auc'] = roc_auc_score(y_true, prob_log)

		return scores


