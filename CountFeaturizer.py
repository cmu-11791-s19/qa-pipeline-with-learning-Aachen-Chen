from Featurizer import Featurizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


#This is a subclass that extends the abstract class Featurizer.
class CountFeaturizer(Featurizer):

	#The abstract method from the base class is implemeted here to return count features
	def getFeatureRepresentation(self, X_train, X_val):
		count_vect = CountVectorizer()
		X_train_counts = count_vect.fit_transform(X_train)
		X_val_counts = count_vect.transform(X_val)
		return X_train_counts, X_val_counts