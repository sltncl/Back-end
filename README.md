# Back-end
 
The Back-end repository is composed as follows:

- Preprocessing directory, contain the dataset and stopwords directories and the Processing.py file
- TrainingTestML directory, contains the MLmodel.py file
- Dataset directory, contains the unprocessed dataset (IMDB_Dataset.csv) and the processed one (New2DataSet.csv);
- Stopwords directory, set of stopwords used to process the dataset;
- main.py, contains the instructions to be able to print the performance of all the models analyzed in our project;
- Pmodel.py, contains the instructions to view the performance of the model saved by the file MLmodelCountvLogisticRegression.py;
- MLmodel.py, contains a class with methods for training, tuning the Machine Learning model;
- Processing.py, contains a class with methods that are used to carry out the pre-processing;
- MLmodelCountvLogisticRegression.py, contains the instructions to be able to train a machine learning model, view the performance and save the trained model in a model.joblib file.

The best performing model is achieved by using the CountVectorizer as the vectorizer and LogisticRegression as the model. 
The file to execute in order to view only the performance of the most precise model is the MLmodelCountvLogisticRegression.py file, once executed a model.joblib file will be created within the repository. 
After the model.joblib file is created, the Pmodel.py file can be executed.



