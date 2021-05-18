from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/test')
def send():
    page = request.args.get(
        'port', default=8000, type = int
    )
    filter = request.args.get(
        'IP', default='localhost', type = str
    )
    print(page)
    print(filter)
    return ''

if __name__ == '__main__':
    app.run(host = 'localhost', port = 8001, debug = True)