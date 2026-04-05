"""SafePlate - dietary trigger detection test lab."""

from flask import Flask

app = Flask(__name__)


@app.route("/")
def home():
    return "SafePlate is running"
