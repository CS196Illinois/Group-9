from flask import Flask
from flask import request

app = Flask(__name__)

@app.route('/update')
def hello_world():
    url = request.args['url']

    return 'Hello, World!'