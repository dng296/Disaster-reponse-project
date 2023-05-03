import sys
import pickle
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import nltk
nltk.download(['punkt','wordnet','stopwords'])
import re
from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords 
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier


def load_data(database_filepath):
    """
    Load cleaned datasset from sqlalchemy database, define feature and target variables X and Y
    Args: database_filepath (.db extension) to be entered as string
    Returns: X and Y as arrays
    """
    # read in file
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('Categorized_messages', con=engine)
    # define features and label arrays
    X = df.message.values
    Y = df.drop(['id','message','original', 'genre'], axis=1).values
    # also extract the category names (column names) for Y
    category_names = df.drop(['id','message','original', 'genre'], axis=1).columns
    return X, Y, category_names


def tokenize(text):
    """
    Text processing steps which include normalizing, removing punctuation, tokenizing, removing stops words, and lemmatizing
    Args: text (X variable)
    Returns: clean tokens
    """
    
    #normalize case and remove punctuation using regular expression
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    #tokenize text
    tokens = word_tokenize(text)
    
    #remove stop words using list comprehension
    stop_words = stopwords.words('english')
    tokens = [tok for tok in tokens if tok not in stop_words]
    
    #initiate lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    #iterate through each token and lemmatize, remove leading/trailing white spaces
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).strip()
        clean_tokens.append(clean_tok)
    return clean_tokens


def build_model():
    """
    Building a pipeline to generate features from text data using CountVectorizer and TfidfTransformer
    then fit it to a RandomForestClassifier model with multiple outputs 
    Implementing a grid search to optimize the model
    Args: none
    Returns: best model obtained from GridSearchCV
 
    """
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {
    'tfidf__use_idf':(True, False),
    'clf__estimator__n_estimators':(10, 20, 30)
    }
    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    # Using the trained model to predict on test data and print out a classification report
    Y_pred = model.predict(X_test)
    print(classification_report(Y_test, Y_pred, zero_division = 0, target_names=category_names))


def save_model(model, model_filepath):
    #export model as a pickle file
    pickle.dump(model, open(model_filepath,'wb'))


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