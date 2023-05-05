import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from plotly.subplots import make_subplots
#from sklearn.externals import joblib
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('Categorized_messages', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # adding graph 1: number of requests for each category
    category_name = df.iloc[:,4:].columns
    category_counts = df.iloc[:,4:].sum()
    sorted_counts, sorted_category = zip(*sorted(zip(category_counts, category_name)))

    #graph 2: group the requests by genre
    cate_genre_count = df[df.columns[3:]].groupby('genre').sum()

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=sorted_counts,
                    y=sorted_category,
                    orientation='h'
                )
            ],

            'layout': {
                'title': 'Counts of reports for each category',
                'xaxis': {'title': 'Number of requests'},
                'yaxis':{'tickangle':'-30'},
                'width':'1200',
                'height':'800'
            }
        },

        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {'title': "Count"},
                'xaxis': {'title': "Genre"}
            }
        },

        {
            'data':[
                Bar(
                    x=cate_genre_count.loc['direct'].sort_values(ascending=False).head(10),
                    y=list(cate_genre_count.loc['direct'].sort_values(ascending=False).head(10).index),
                    name='Direct',
                    orientation='h',
                )
            ],

            'layout':{
                'title': 'Top 10 reported categories via direct source',
                #'barmode':'group',
                'xaxis': {'title': 'Counts'},
                'yaxis':{'tickangle':'-30'},
                'width':'600',
                'height':'600'
            }
        },

        {
            'data':[
                Bar(
                    x=cate_genre_count.loc['news'].sort_values(ascending=False).head(10),
                    y=list(cate_genre_count.loc['news'].sort_values(ascending=False).head(10).index),
                    name='News',
                    orientation='h',
                )
            ],

            'layout':{
                'title': 'Top 10 reported categories via news source',
                #'barmode':'group',
                'xaxis': {'title': 'Counts'},
                'yaxis':{'tickangle':'-30'},
                'width':'600',
                'height':'600'
            }
        },

        {
            'data':[
                Bar(
                    x=cate_genre_count.loc['social'].sort_values(ascending=False).head(10),
                    y=list(cate_genre_count.loc['social'].sort_values(ascending=False).head(10).index),
                    name='Social',
                    orientation='h',
                )
            ],

            'layout':{
                'title': 'Top 10 reported categories via social media',
                #'barmode':'group',
                'xaxis': {'title': 'Counts'},
                'yaxis':{'tickangle':'-30'},
                'width':'600',
                'height':'600'
            }
        },
    ]



    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
