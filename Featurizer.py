import abc
from abc import abstractmethod

class Featurizer:
	__metaclass__ = abc.ABCMeta
	@classmethod
	def __init__(self): #constructor for the abstract class
		pass

	#This is the abstract method that is implemented by the subclasses.
	@abstractmethod
	def getFeatureRepresentation(self, X_train, X_val):
		pass

