from Classifier import Classifier
from sklearn.naive_bayes import MultinomialNB


#This is a subclass that extends the abstract class Classifier.
class MultinomialNaiveBayes(Classifier):

	#The abstract method from the base class is implemeted here to return multinomial naive bayes classifier
	def buildClassifier(self, X_features, Y_train):
		clf = MultinomialNB().fit(X_features, Y_train)
		return clf