# Back-end
 
La repository Back-end è così composta:

- Directory Dataset, sono presenti il csv non processato e quello processato;
- Directory Stopwords, insieme delle stopwords utilizzate per processare il dataset;
- main.py, contiene le istruzioni per poter visualizzare le prestazione di un modello serializzato;
- MLmodel.py, classe per training, tuning del modello di Machine Learning;
- Processing.py, classe per Preprocessing del dataset originale;
- MLmodelCountvLogisticRegression.py, contiene le istruzioni per poter allenare un modello di machine learning, visualizzare le prestazioni e salvare il modello in un file model.joblib.

Il tipo di vettorizzatore utilizzato è il CountVectorizer e il modello di machine learning utilizzato è LogisticRegression.
Per poter vedere le prestazioni del modello allenato eseguire il file main.py.




