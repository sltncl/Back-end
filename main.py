import joblib

# Import the trained model
model = joblib.load('model.joblib')

print('\nCOUNT VECTORIZER')
print('Logistic Regression\n')

model.PrintSummaryOfTraining()
