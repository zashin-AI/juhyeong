from flask import Flask, render_template
from flask.globals import request
from werkzeug.utils import secure_filename # 이전 버전에서는 werkzeug 뒤에 utils 없이 사용했음

app = Flask(__name__)

# upload html file rendering
@app.route('/')
def render_file():
    return render_template('upload.html')

# file upload
@app.route('/fileUpload', methods = ['GET', 'POST'])
def upload_file():
    if request.method == 'POST': # POST 형식으로 요청이 들어오는 경우에 실행
        f = request.files['file']
        # save dir + filename(secure_filename)
        f.save('c:/nmb/nmb_data/web/' + secure_filename(f.filename))
        return 'uploads 디렉토리 > 파일 업로드 성공'

if __name__ == '__main__':
    app.run(debug=True)