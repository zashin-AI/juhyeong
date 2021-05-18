from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return 'home'

@app.route('/send')
def send():
    return render_template('/UI.html', data = str(123))

if __name__ == '__main__':
    app.run()