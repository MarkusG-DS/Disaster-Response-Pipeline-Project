import sys
import pandas as pd
import numpy as np
import re

from sqlalchemy import create_engine
import pickle

import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, make_scorer, accuracy_score, f1_score, fbeta_score

from sklearn.model_selection import GridSearchCV

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

#
class StartingVerbExtractor(BaseEstimator, TransformerMixin):

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)

#
def load_data(database_filepath, database_filename):
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table(database_filename, engine)
    
    df = df.drop(['child_alone'],axis=1)
    
    X = df['message']
    y = df.iloc[:,4:]
    category_names = y.columns
    
    return X, y, category_names


def tokenize(text):
    
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    stop_words = stopwords.words("english")
    
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    # Extract and replace all urls from text 
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
        
    clean_tokens= [lemmatizer.lemmatize(word).lower().strip() for word in tokens]
    
    return clean_tokens


def build_model():
    # text processing and model pipeline
    model = []
    model = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),
            ('starting_verb', StartingVerbExtractor())        
        ])),
        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])
     
        
    # define parameters for GridSearchCV

#parameters = {
      #  'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
      #  'features__text_pipeline__vect__max_df': (0.5, 0.75, 1.0),
      #  'features__text_pipeline__vect__max_features': (None, 5000, 10000),
      #  'features__text_pipeline__tfidf__use_idf': (True, False),
      #  'clf__n_estimators': [50, 100, 200],
      #  'clf__n_estimators': [50, 100]
      #  'clf__min_samples_split': [2, 3, 4],
      #  'features__transformer_weights': (
      #      {'text_pipeline': 1, 'starting_verb': 0.5},
      #      {'text_pipeline': 0.5, 'starting_verb': 1},
      #      {'text_pipeline': 0.8, 'starting_verb': 1},
      # )
#    }
    
    # create gridsearch object and return as final model pipeline    
    #cv = GridSearchCV(model, param_grid=parameters, scoring='f1_micro', verbose = 2, n_jobs = -1)
        
    return model
    
        


def evaluate_model(model, X_test, Y_test, category_names):
        y_pred = model.predict(X_test)
        df_y_pred = pd.DataFrame(y_pred, columns = Y_test.columns)
        
        accuracy = ((y_pred == Y_test).mean()).mean()
        print('average overall accuracy {0:.2f}% \n'.format(accuracy*100))
        
        for column in df_y_pred.columns:
            print('/n')
            print('Column: {}\n'.format(column))
            print(classification_report(Y_test[column], df_y_pred[column]))
               


def save_model(model, model_filepath):
        pickle.dump(model, open(model_filepath, 'wb'))

        
def main():
    if len(sys.argv) == 4:
        database_filepath, database_filename, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath, database_filename)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
