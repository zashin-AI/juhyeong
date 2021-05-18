from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return 'home'

@app.route('/send')
def send():
    return render_template('/UI.html', data = 'c:/nmb/nada/answer.txt')

if __name__ == '__main__':
    app.run()