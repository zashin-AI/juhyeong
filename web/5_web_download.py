# 다운로드 방식에는 두가지가 있다.
# 하나는 file 을 stream 으로 만들어 다운 받는 방식 (서버에 파일이 저장 되지 않음)
# 두 번째는 static file 로 만들어서 그 파일을 다운로드 하는 방식

# from flask import Flask, render_template, send_file, request

# app = Flask(__name__)

# @app.route('/')
# def home():
#     return render_template('/upload.html')

# @app.route('/fileUpload', methods = ['GET', 'POST'])
# def upload_file():
#     if request.method == 'POST': # POST 형식으로 요청이 들어오는 경우에 실행
#         f = request.files['file']
#         # save dir + filename(secure_filename)
#         f.save('c:/nmb/nmb_data/web/' + secure_filename(f.filename))
#         return 'uploads 디렉토리 > 파일 업로드 성공'

# @app.route('/download_txt_file')
# def wav_file_download_with_file():
#     file_name = f"static/results/file_path.txt"
#     return send_file(
#         file_name,
#         mimetype='text/txt',
#         attachment_filename='download_txt_file.txt',
#         as_attachment=True
#     )

# if __name__ == '__main__':
#     app.run()

from flask import Flask, send_file, render_template, request
from werkzeug.utils import secure_filename

app = Flask(__name__, static_url_path='/static')

@app.route('/')
def home():
    return render_template(
        'upload.html'
    )

@app.route('/uploadFile', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        f.save('c:/nmb/nmb_data/web/' + secure_filename(f.filename))
        return render_template('download.html')

@app.route('/download')
def download_file():
    p = 'c:/nmb/nmb_data/web/answer.txt'
    return send_file(p, as_attachment=True)

@app.route('/read')
def read_txt():
    f = open('c:/nmb/nada/web/static/answer.txt', 'r', encoding='utf-8')
    return "</br>".join(f.readlines())

if __name__ == '__main__':
    app.run(debug=True, threaded = True)