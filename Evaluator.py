import abc
from abc import abstractmethod
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score

class Evaluator:
	__metaclass__ = abc.ABCMeta
	@classmethod
	def __init__(self): #constructor for the abstract class
		pass

	#This is a class method that gets accuracy of the model
	@classmethod
	def getAccuracy(self, Y_true, Y_pred):
		accuracy = accuracy_score(Y_true, Y_pred)
		return accuracy
	
	#This is a class method that gets precision, recall and f-measure of the model	
	@classmethod
	def getPRF(self, Y_true, Y_pred):
		prf = precision_recall_fscore_support(Y_true, Y_pred, average='micro')
		precision = prf[0]
		recall = prf[1]
		f_measure = prf[2]
		return precision, recall, f_measure
