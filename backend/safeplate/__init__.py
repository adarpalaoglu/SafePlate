from flask import Flask

# This creates the main Flask application object.
# Think of this as the "brain" of your website.
app = Flask(__name__)

# This defines what happens when someone goes to the home page: "/".
@app.route("/")
def home():
    # This text will be shown in the browser.
    return "SafePlate is running ✅"
