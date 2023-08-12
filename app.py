import sqlite3

import os.path

from flask import Flask, render_template, url_for, request, flash, session, redirect, abort, g

from main import action

models, data = action()

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


@app.route("/index", methods=["POST", "GET"])
@app.route("/", methods=["POST", "GET"])
def index():
    db = get_db()
    dbase = FDataBase(db)

    sentiment = "=D"
    rating = "=D"

    # if data came properly
    if request.method == "POST":

        if len(request.form['post']) > 10:

            text = request.form['post']

            pad = data['data_senti'].preprocess(text)
            sentiment = models["gru"].predict(pad)

            if sentiment == 0:
                pad = data['data_0'].preprocess(text)
                rating = models["gru_0"].predict(pad)

            elif sentiment == 1:
                pad = data['data_1'].preprocess(text)
                rating = models["gru_1"].predict(pad)

            else:
                pad = data['data_2'].preprocess(text)
                rating = models["gru_2"].predict(pad)

            flash(f"Sentiment: {sentiment}", category='success')
            flash(f"Rating: {rating}", category='success')
        else:
            flash("Too short", category='error')
    return render_template("index.html", sentiment=sentiment, rating=rating)



@app.route("/post/<alias>")
def showPost(alias):
    db = get_db()
    dbase = FDataBase(db)
    title, post = dbase.getPost(alias)
    if not title:
        abort(404)
    print(alias)
    return render_template('post.html', menu=dbase.getMenu(), title=title, post=post)


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
