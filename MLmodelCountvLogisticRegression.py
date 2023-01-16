# import of all library we will need
import joblib
import pandas as pd
from TrainingTestML import MLmodel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# Import the new dataset from New2DataSet.csv
df = pd.read_csv('PreProcessing/Dataset/New2DataSet.csv')
# Set the features and label
# A feature is an input variable
# A label is the thing we're predicting
X = df['review']
y = df['sentiment']

# I will use a count vectorizer to vectorize the text data in the review column
# and then use three different classification models from scikit-learn models.

'''
# The CountVectorizer uses the bag of word approach that ignores the text structures and only extract information from the word counts. 
# It will transform each document into a vector. 
# The inputs of the vector are the occurrence count of each unique word for this document. 
# When having m documents in the corpus, and there are n unique words from all m documents, the CountVectorizer will transform the text data into a m*n sparse matrix.
'''

print('\nCOUNT VECTORIZER')
print('\nLogistic Regression')
# Defines the variable countVectlr as an object Vectorizer
model = MLmodel.MLmodel(CountVectorizer, LogisticRegression(max_iter=500,),X,y)
model.trainingModel()
model.PrintSummaryOfTraining()
# we store the trained model, the preprocessor and the vectorizer in 3 different memory location files
joblib.dump(model, "model.joblib")

