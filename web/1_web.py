from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello():
    return 'hello world!'

@app.route('/home')
def home():
    return 'home'

if __name__ == '__main__':
    app.run()