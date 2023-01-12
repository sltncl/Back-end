# Back-end
 
La repository Back-end è così composta:

- Directory Dataset, sono presenti il csv non processato e quello processato;
- Directory Stopwords, insieme delle stopwords utilizzate per processare il dataset;
- main.py, contiene le istruzioni per poter printare le prestazioni di tutti i modelli analizzati nel nostro progetto;
- Pmodel.py, contiene le istruzioni per poter visualizzare le prestazione del modello salvato dal file MLmodelCountvLogisticRegression.py;
- MLmodel.py, classe per training, tuning del modello di Machine Learning;
- Processing.py, classe per Preprocessing del dataset originale;
- MLmodelCountvLogisticRegression.py, contiene le istruzioni per poter allenare un modello di machine learning, visualizzare le prestazioni e salvare il modello allenato in un file model.joblib.

Il modello con prestazioni più elevate si ottiene utilizzando il CountVectorizer come vettorizzatore e LogisticRegression come modello.
Il file da eseguire per poter visualizzare solo le prestazioni del modello più preciso è il file MLmodelCountvLogisticRegression.py, una volta eseguito verrà creato all'interno della repository un file model.joblib.
Dopo la creazione del file model.joblib potrà essere eseguito il file Pmodel.py.




