import abc
from abc import abstractmethod

class Retrieval:
	__metaclass__ = abc.ABCMeta
	@classmethod
	def __init__(self): #constructor for the abstract class
		pass

	@classmethod
	def getLongSnippets(self, question):
		longSnippets = question['contexts']['long_snippets']
		fullLongSnippets = ' '.join(longSnippets)
		return fullLongSnippets


	@classmethod
	def getShortSnippets(self, question):
		shortSnippets = question['contexts']['short_snippets']
		fullShortSnippets = ' '.join(shortSnippets)
		return fullShortSnippets