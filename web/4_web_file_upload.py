from flask import Flask, render_template
from flask.globals import request
from werkzeug.utils import secure_filename

app = Flask(__name__)

# upload html file rendering
@app.route('/')
def render_file():
    return render_template('upload.html')

# file upload
@app.route('/fileUpload', methods = ['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        # save dir + filename
        f.save(secure_filename(f.filename))
        return 'uploads 디렉토리 > 파일 업로드 성공'

if __name__ == '__main__':
    app.run(debug=True)