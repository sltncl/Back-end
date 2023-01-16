# import of all library we will need
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import GridSearchCV
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer

# we define the class that provides the vectoring
class MLmodel:
    def __init__(self, typeVectorizer, model, features, label):
        # the type of chosen vector
        self.typeVectorizer = typeVectorizer
        self.features = features
        self.label = label
        # create a vectorizer
        self.cv = self.getVectorizer()
        # the type of chosen ML model
        self.model = model
        # splitting of dataset in train-test
        self.X_train, self.X_test, self.y_train, self.y_test = self.ttSplit()

    # method that provides splitting
    def ttSplit(self):
        # The fit(data) method is used to compute the mean and std dev for a given feature ,
        # to be used further for scaling.
        # The transform(data) method is used to perform scaling with mean and std dev calculated using the.fit() method.
        # The fit_transform() method does both fits and transform.
        self.features = self.cv.fit_transform(self.features)
        # Split the feature and label into random train and test subsets.
        # There are 4 subsets: X_train, X_test, y_train, y_test
        # Test size represents the absolute number of test samples.
        # The train size value is set to the complement of the test size.
        X_train, X_test, y_train, y_test = train_test_split(self.features, self.label,
                                                            test_size=0.2, random_state=24)
        return X_train, X_test, y_train, y_test

    # initialization of chosen vector
    def getVectorizer(self):
        cv = self.typeVectorizer(stop_words='english', ngram_range=(1, 2))
        return cv

    # method that provides the training of model
    def trainingModel(self):
        # Definition of the model used
        model = self.model
        # Training the model
        model.fit(self.X_train, self.y_train)

    # method that try to calculate if the string have a positive or negative sentiment and with what level of confidence
    def predict(self, vtext):
        # we take the already preprocessed and vectorized string and return prediction and his level of confidence
        return self.model.predict(vtext), self.model.predict_proba(vtext)

    # this method provides the printing of summary
    def PrintSummaryOfTraining(self):
        p_train = self.model.predict(self.X_train)
        p_test = self.model.predict(self.X_test)
        # Accuracy score
        acc_train = accuracy_score(self.y_train, p_train)
        acc_test = accuracy_score(self.y_test, p_test)
        print(f'Train acc. {acc_train}, Test acc. {acc_test}')
        # Confusion matrix
        #                | Positive Prediction | Negative Prediction
        # Positive Class | True Positive (TP)  | False Negative (FN)
        # Negative Class | False Positive (FP) | True Negative (TN)
        cm_lr = confusion_matrix(self.y_test, p_test)
        tn, fp, fn, tp = confusion_matrix(self.y_test, p_test).ravel()
        print(f'TRUE POSITIVE: {tp}')
        print(f'TRUE NEGATIVE: {tn}')
        print(f'FALSE POSITIVE: {fp}')
        print(f'FALSE NEGATIVE: {fn}')
        # True positive and true negative rates
        tpr_lr = round(tp / (tp + fn), 4)
        tnr_lr = round(tn / (tn + fp), 4)
        print(f'TRUE POSITIVE RATES: {tpr_lr}')
        print(f'TRUE NEGATIVE RATES: {tnr_lr}')
        # F1Score
        # Changing the pos_label value default in the f1_score() to consider positive scenarios when 'positive' appears
        f1score_train = f1_score(self.y_train, p_train, pos_label='positive')
        f1score_test = f1_score(self.y_test, p_test, pos_label='positive')
        print(f'TRAIN F1 SCORE: {f1score_train}, TEST F1 SCORE {f1score_test}')
        # Precision Score
        # Changing the pos_label value default in the precision_score() to consider positive scenarios when 'positive' appears
        precisionscore_train = precision_score(self.y_train, p_train, pos_label='positive')
        precisionscore_test = precision_score(self.y_test, p_test, pos_label='positive')
        print(f'TRAIN PRECISION SCORE: {precisionscore_train}, TEST PRECISION SCORE:  {precisionscore_test}')
        # Recall score
        # Changing the pos_label value default in the recall_score() to consider positive scenarios when 'positive' appears
        recallscore_train = recall_score(self.y_train, p_train, pos_label='positive')
        recallscore_test = recall_score(self.y_test, p_test, pos_label='positive')
        print(f'TRAIN RECALL SCORE:  {recallscore_train}, TEST RECALL SCORE:  {recallscore_test}')

    # This method provides the tuning of hyperparameters for the chosen model,passed as a parameter without parentheses,
    # It's required even the type of the optymization as "Bayesian" or as "Grid Search"
    def tuningHyperparameters(self, model, type):
        if type == "Bayesian":
            # Applying Bayesian optymization to find the best model and the best parameters
            # Hyperparameters
            if model == SVC:
                parameters = {"C": Real(1e-6, 1e+6, prior='log-uniform'),
                              'gamma': Real(1e-6, 1e+1, prior='log-uniform'),
                              'degree': Integer(1, 8), 'kernel': Categorical(['linear', 'poly', 'rbf']), }
            elif model == KNeighborsClassifier:
                parameters = {'n_neighbors': [5, 7, 9, 11, 13, 15],
                              'weights': ['uniform', 'distance'],
                              'metric': ['minkowski', 'euclidean', 'manhattan']}
            elif model == LogisticRegression:
                parameters = {'penalty': ['none', 'l2'],
                              'solver': ['newton-cg', 'lbfgs', 'sag', 'saga'],
                              'multi_class': ['auto', 'ovr', 'multinomial']}
            # Bayesian Search
            classifier = BayesSearchCV(model(), parameters, cv=5, n_jobs=-1)
        elif type == "Grid Search":
            # Applying Grid Search to find the best model and the best parameters
            # Hyperparameters
            if model == SVC:
                parameters = {'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                              'C': np.arange(1, 20)}
            elif model == KNeighborsClassifier:
                parameters = {'n_neighbors': range(1, 30, 2),  # range(start, stop, step)
                              'weights': ['uniform', 'distance'],
                              'metric': ['euclidean', 'manhattan', 'minkowski'],
                              'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                              'leaf_size': range(1, 50, 5)}
            elif model == LogisticRegression:
                parameters = {'penalty': ['l1', 'l2', 'elasticnet', 'none'],
                              'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                              'multi_class': ['auto', 'ovr', 'multinomial']}
            # Grid Search
            classifier = GridSearchCV(model(), parameters, cv=5, n_jobs=-1)
        # regardless the type of the optymization chosen we execute this section
        # Fitting the data to our model
        classifier.fit(self.X_train, self.y_train)
        # Best parameters
        bestParameters = classifier.best_params_
        print(bestParameters)
        # Highest accuracy
        highestAccuracy = classifier.best_score_
        print(highestAccuracy)
