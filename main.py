from flask import Flask, render_template
from flaskext.markdown import Markdown

app = Flask(__name__)
Markdown(app)


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/data_processing")
def data_processing():
    return render_template("data_process.m.html")


@app.route("/exploratory")
def exploratory():
    return render_template("exploratory.html")


@app.route("/model")
def model_code():
    return render_template("model.m.html")


@app.route("/model_notebook")
def model_notebook():
    return render_template("lstm.html")


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
