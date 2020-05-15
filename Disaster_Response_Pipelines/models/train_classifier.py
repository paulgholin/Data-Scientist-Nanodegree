import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import fbeta_score, make_scorer, accuracy_score
from typing import Tuple, List
import nltk
import pickle
nltk.download('punkt')
nltk.download('wordnet')

# ignore all future warnings
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

def load_data(database_filepath):
    '''
    Load the database into Pandas DataFrames.
    Args: database_filepath: the path to the database.
    Returns: X: features (messages).
             y: categories (one-hot encoded).
             An ordered list of categories.
    '''
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql("SELECT * FROM disaster_response", engine) 
    X = df['message']
    y = df.drop(['id', 'message', 'original', 'genre'],  axis=1).astype(float)
    categories = y.columns.values
    return X, y, categories

def tokenize(sentense):
    '''
    Tokenize string.
    Args: text string.
    Returns: A list of tokens.
    '''
    tokens = nltk.word_tokenize(sentense)
    lemmatizer = nltk.WordNetLemmatizer()
    return [lemmatizer.lemmatize(x).lower().strip() for x in tokens]

def build_model():
    '''
    Build pipeline and GridSearch.
    Args: None.
    Returns: Model after GridSearch.
    '''
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])
    
    parameters = {
    'clf__estimator__learning_rate': [0.5, 1],
    'clf__estimator__n_estimators': [50, 100]
    }

    cv = GridSearchCV(pipeline, parameters, n_jobs =-1)
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluate model by printing a classification report.
    Args: Model, features, labels, and a list of categories.
    Returns: None.
    '''
    y_pred = model.predict(X_test)
    print(classification_report(Y_test, y_pred, target_names=category_names))
    for i, cat in enumerate(Y_test.columns.values):
        print(f"{cat}: {accuracy_score(Y_test.values[:, i], y_pred[:, i])}")
    print(f"Accuracy = {accuracy_score(Y_test, y_pred)}")

def save_model(model, model_filepath):
    '''
    Save model using pickle.
    Args: Model, filepath to save.
    Returns: None.
    '''
    with open(model_filepath, 'wb') as file:  
        pickle.dump(model, file)

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
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