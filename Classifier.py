import abc
from abc import abstractmethod

class Classifier:
	__metaclass__ = abc.ABCMeta
	@classmethod
	def __init__(self): #constructor for the abstract class
		pass

	#This is the abstract method that is implemented by the subclasses.
	@abstractmethod
	def buildClassifier(self, X_features, Y_train):
		pass
