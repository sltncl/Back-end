# import of all library we will need
import pandas as pd
import spacy
import re
import csv


# we define the class that provides the preprocessing
class DataProcessing:
    def __init__(self):
        # core = language for spacy,disable = pipeline phase that you want to deactivate
        self.core, self.disable = self.setCoreandDisable()
        # Defines nlp with core and disabled passed in the constructor
        self.nlp = self.getInitNlp(self.core, self.disable)
        # we initialize the stopwords
        self.stopwords = self.Stopwords()

    # expand the stopwords
    def Stopwords(self):
        # initialize the stopword
        with open("./Stopwords/Mesi_festività.txt", 'r') as s:
            months = s.readlines()
        with open("./Stopwords/giorni_settimana.txt", 'r') as s:
            days = s.readlines()
        with open("./Stopwords/Nomi.txt", 'r') as s:
            names = s.readlines()
        with open("./Stopwords/Stopword.txt", 'r') as s:
            stopwords = s.readlines()
        # combine all stopwords
        [stopwords.append(x.rstrip()) for x in months]
        [stopwords.append(x.rstrip()) for x in days]
        [stopwords.append(x.rstrip()) for x in names]
        # set the stopwords to lower case
        stopwords = [stopW.lower() for stopW in stopwords]
        return stopwords

    # read the csv
    def readOldCSV(self, oldPath):
        # Returns the reading of the csv file passed in the constructor
        return pd.read_csv(oldPath)

    # initialization of nlp
    def getInitNlp(self, core, disable):
        # Initialize spacy ‘en’ model, keeping only component needed for lemmization and creating an engine
        return spacy.load(core, disable=disable)

    def setCoreandDisable(self):
        core = 'en_core_web_sm'
        disable = ['parser', 'ner']
        return core, disable

    def setRegex(self):
        # set three regex compiled once and used multiple times.
        # It's setted a regex for the special caracter, a regex for the web's site and a regex for the email
        self.caratteri_speciali = re.compile(r'[^A-Za-z@]')
        self.siti_web = re.compile(r'\S+com')
        self.email = re.compile(r'\S+@\S+')

    # It does data processing
    def text_preprocessing(self, text):
        # we initialize the regex
        self.setRegex()
        # string tokenizzation and lemmization with nlp(from spicy)'s methods and delete of punct
        words = [token.lemma_ for token in self.nlp(text) if not token.is_punct]
        # filt with the regex
        words = [re.sub(self.caratteri_speciali, ' ', word) for word in words]
        words = [re.sub(self.siti_web, ' ', word) for word in words]
        words = [re.sub(self.email, ' ', word) for word in words]
        # delete blanck
        words = [word for word in words if word != ' ' if word != '']
        # filt with the stopwords
        words = [word.lower() for word in words if word.lower() not in self.stopwords]
        # combine a list into one string
        string = " ".join(words)
        # replaces n empty spaces with only one
        string = " ".join(string.split())
        return string

    # this creates a new dataset, preprocessed, starting from an old one
    def createNewDataSet(self, oldpath, newPath):
        self.newPath = newPath
        # Defines the dataframe to use passed in the constructor parameters
        self.df = self.readOldCSV(oldpath)
        # delete the duplicates and the null values
        self.df.dropna()
        self.df.drop_duplicates()
        count = 0
        # Declare a .csv file in write mode.
        # Pass the column names in the csv.DictWriter() method with the file to save.
        # This method returns a writer with which we can add the desired values row by row.
        with open(self.newPath, 'w') as csv_file:
            columnsname = ['review', 'sentiment']
            writer = csv.DictWriter(csv_file, fieldnames=columnsname)
            writer.writeheader()
            # Iterates on original dataset and modifies the new one that will contain the reviews processed
            for index in self.df['review']:
                textReview = self.df.iloc[count]['review']
                textSentiment = self.df.iloc[count]['sentiment']
                writer.writerow({'review': self.text_preprocessing(textReview), 'sentiment': textSentiment})
                count = count + 1
