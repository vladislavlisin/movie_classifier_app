import os.path
import sqlite3
import os
import pickle
import pandas as pd
import numpy as np
import os.path

import fasttext
import fasttext.util

from flask import Flask, render_template, url_for, request, flash, session, redirect, abort, g
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

from model import model
from data import Data


# WSGI app


# sqlite: one writes - many read, min but full set of instructs

# configuration
DATABASE = '/tmp/flsite.db'
DEBUG = True
SECRET_KEY = 'rivrbeivbruinvorniwvnioeo23ui4728'

from FDataBase import FDataBase


app = Flask(__name__)
app.config.from_object(__name__)
#app.config["SECRET_KEY"] = 'rivrbeivbruinvorniwvnioeo23ui4728'


app.config.update(dict(DATABASE=os.path.join(app.root_path, 'flsite.db')))


def connect_db():
    conn = sqlite3.connect(app.config['DATABASE'])
    conn.row_factory = sqlite3.Row
    return conn


def create_db():
    db = connect_db()
    with app.open_resource('sq_db.sql', mode='r') as f:
        db.cursor().executescript(f.read())
    db.commit()
    db.close()

# how to run create_db without running all web server - use  console

# how to use bd - connection to bd in requires

menu = [{"name": 'installing', "url": "install-flask"},
        {"name": "the first app", "url": "first-app"},
        {"name": "feedback", "url": "contact"}]

def get_db():
    # when we got require app context is created, g - contain user info
    # check: do g has a property 'link_db'
    if not hasattr(g, 'link_db'):
        g.link_db = connect_db()
    return g.link_db


####################################################
#                   GET FLASK                      #
####################################################

@app.route("/add_post", methods=["POST", "GET"])
def addPost():
    db = get_db()
    dbase = FDataBase(db)

    # if data came properly
    if request.method == "POST":
        # some simple checks for a title and a post
        if len(request.form["name"]) > 4 and len(request.form['post']) > 10:
            res = dbase.addPost(request.form['name'], request.form['post'], request.form['url'])
            if not res:
                flash("Add post error", category='error')
            else:
                flash("Post was successfully added", category='success')
        else:
            flash("Add post error", category='error')
    return render_template("add_post.html", menu=dbase.getMenu(), title='Add post')



@app.route("/post/<alias>")
def showPost(alias):
    db = get_db()
    dbase = FDataBase(db)
    title, post = dbase.getPost(alias)
    if not title:
        abort(404)
    print(alias)
    return render_template('post.html', menu=dbase.getMenu(), title=title, post=post)


@app.route("/index")
@app.route("/")
def index():
    db = get_db()
    dbase = FDataBase(db)
    print(url_for("index"))
    return render_template('index.html', menu=dbase.getMenu(), posts=dbase.getPostsAnonce())


@app.teardown_appcontext
def close_db(error):
    if hasattr(g, "link_db"):
        g.link_db.close()


@app.route("/about")
def about():
    print(url_for("about"))
    return render_template('about.html', title="About site", menu=menu)

@app.route("/contact", methods=['POST', 'GET'])
def contact():

    if request.method == "POST":
        if len(request.form["username"]) > 2:
            flash("message has been sent", category="success")
        else:
            flash("error", category="error")
        print(request.form)

    return render_template("contact.html", title="feedback", menu=menu)

# proceed 404 error
@app.errorhandler(404)
def pageNotFound(error):
    return render_template('page404.html', title="page hasn't been found", menu=menu), 404


@app.route("/login", methods=["POST", "GET"])
def login():
    if "userLogged" in session:
        return redirect(url_for('profile', username=session['userLogged']))
    elif request.method == "POST" and request.form['username'] == 'vladfoxin' and request.form['psw'] == "123":
        session['userLogged'] = request.form['username']
        return redirect(url_for('profile', username=session['userLogged']))

    return render_template('login.html', title='autorithation', menu=menu)

@app.route('/profile/<username>')
def profile(username):

    if 'userLogged' not in session or session['userLogged'] != username:
        abort(401)

    return f"user's profile: {username}"

###############################################################################
#                                ALGO PART                                    #
###############################################################################


def get_fasttext_model():

    #file_path = './cc.en.300.bin'
    #if not os.path.exists(file_path):
    fasttext.util.download_model('en', if_exists='ignore')
    ft = fasttext.load_model('cc.en.300.bin')

    return ft


# преобразуем последовательности чисел в плотные векторные представления FT

def get_ft_embed_matrix(train_word_index, model, EMBEDDING_DIM=300):

    train_embedding_weights = np.zeros((len(train_word_index), EMBEDDING_DIM))

    for word, index in train_word_index.items():
        if word in model:
            train_embedding_weights[index,:] = model[word]
        else:
            np.random.rand(EMBEDDING_DIM)
    return train_embedding_weights

def pipeline(X, model_name, data_title, ft, labels_num):

    X.fit()
    X.preprocess()

    X_train, X_test, y_train, y_test = train_test_split(X.pad_cut_seq,
                                                        X.y,
                                                        train_size=0.80,
                                                        test_size=0.20,
                                                        random_state=1,
                                                        stratify=X.y)

    ru_ft_emb = get_ft_embed_matrix(X.tokenizer.word_index, ft, EMBEDDING_DIM=300)
    input_length = X_train.shape[1]

    gru = model(input_length, embeddings=ru_ft_emb, trainable=False, labels_num=labels_num, vocab_size=None)

    gru.train(X_train, y_train, model_name=model_name, epochs=20, batch_size=32)

    model_path = "./models/" + model_name
    data_path =  "./data/" + data_title
    embeddings_path = "./data/ft_" + data_title

    with open(embeddings_path, "wb") as fp:  # Pickling
        pickle.dump(ru_ft_emb, fp)

    with open(model_path, "wb") as fp:  # Pickling
        pickle.dump(gru, fp)

    with open(data_path, "wb") as fp:  # Pickling
        pickle.dump(X, fp)

    return gru

def process_request(text, models):

    sentiment = models["gru"].predict(text)

    if sentiment == 0:
        rating = models["gru_0"].predict(text)

    elif sentiment == 1:
        rating = models["gru_1"].predict(text)

    else:
        rating = models["gru_2"].predict(text)

    print(rating, sentiment)


def input():

    df = pd.read_csv("/content/drive/MyDrive/тестовое задание сервис оценки фильмов/quarter_dataset")

    # prepare data for sentiment model

    y_sentiment = df.Sentiment
    review = df.Review

    idx2label_senti = {
        0: "Negative",
        1: "Positive",
        2: "Neutral",
    }
    label2idx_senti = {
        "Negative": 0,
        "Positive": 1,
        "Neutral": 2
    }

    temp = to_categorical(y_sentiment, num_classes=3)
    x = Data(review, temp, label2idx_senti, idx2label_senti)

    # prepare data for rating models

    data_1_4 = df.loc[df["Rating"] < 5]
    data_5_6 = df.loc[df.Rating.isin([5, 6])]
    data_7_10 = df.loc[df.Rating > 6]

    label2idx_7_10 = {7: 0, 8: 1, 9: 2, 10: 3}
    idx2label_7_10 = {0: 7, 1: 8, 2: 9, 3: 10}

    label2idx_5_6 = {5: 0, 6: 1}
    idx2label_5_6 = {0: 5, 1: 6}

    label2idx_1_4 = {1: 0, 2: 1, 3: 2, 4: 3}
    idx2label_1_4 = {0: 1, 1: 2, 2: 3, 3: 4}

    temp_5_6 = data_5_6.Rating.map(label2idx_5_6)
    temp_7_10 = to_categorical(data_7_10.Rating, num_classes=4)
    temp_1_4 = to_categorical(data_1_4.Rating, num_classes=4)

    # Positive - 1
    x_7_10 = Data(data_7_10.Review, temp_7_10, label2idx_7_10, idx2label_7_10)
    # Neutral - 2
    x_5_6 = Data(data_5_6.Review, temp_5_6, label2idx_5_6, idx2label_5_6)
    # Negative - 0
    x_1_4 = Data(data_1_4.Review, temp_1_4, label2idx_1_4, idx2label_1_4)

    data = [x, x_1_4, x_7_10, x_5_6]

    return data


def passive(model_name, models):

    with open("./models/" + model_name[0], "rb") as fp:  # Unpickling
        models['gru'] = pickle.load(fp)

    with open("./models/" + model_name[1], "rb") as fp:  # Unpickling
        models["gru_0"] = pickle.load(fp)

    with open("./models/" + model_name[2], "rb") as fp:  # Unpickling
        models["gru_1"] = pickle.load(fp)

    with open("./models/" + model_name[3], "rb") as fp:  # Unpickling
        models["gru_2"] = pickle.load(fp)

    return 0


def action(model_name, models):

    data = input()

    ft = get_fasttext_model()

    data_title = ["data_senti", "data_0", "data_1", "data_2"]
    labels = [3, 4, 4, 2]

    # обучаем 4 модели
    for i, j, k, l in zip(data, model_name, data_title, labels):

        models[model_name] = pipeline(i, j, k, ft, l)

    return 0



def main():
    models = {}
    model_name = ["gru", "gru_0", "gru_1", "gru_2"]

    #passive(model_name, models)
    action(model_name, models)

    app.run(debug=True, host='0.0.0.0', port=5000)


if __name__ == "__main__":
    action()