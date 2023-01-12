# Back-end
 
La repository Back-end è così composta:

- Directory Dataset, sono presenti il csv non processato e quello processato;
- Directory Stopwords, insieme delle stopwords utilizzate per processare il dataset;
- main.py, contiene le istruzioni per poter visualizzare le prestazione del modello salvato dal file MLmodelCountvLogisticRegression.py;
- MLmodel.py, classe per training, tuning del modello di Machine Learning;
- Processing.py, classe per Preprocessing del dataset originale;
- MLmodelCountvLogisticRegression.py, contiene le istruzioni per poter allenare un modello di machine learning, visualizzare le prestazioni e salvare il modello allenato in un file model.joblib.

Il tipo di vettorizzatore utilizzato è il CountVectorizer e il modello di machine learning utilizzato è LogisticRegression.
Il file da eseguire è il file MLmodelCountvLogisticRegression.py, una volta eseguito sarà creato all'interno della repository del progetto un file model.joblib.
Dopo la creazione del file model.joblib potrà essere eseguito il file main.py




