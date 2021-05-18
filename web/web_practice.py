from flask import Flask, template_rendered

app = Flask(__name__)

@app.route('/')
def home():
    return 'home'

@app.route('/ui')
def ui():
    return template_rendered('/ui_practice.hitml')

if __name__ == '__main__':
    app.run()