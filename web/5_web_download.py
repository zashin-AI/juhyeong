# 다운로드 방식에는 두가지가 있다.
# 하나는 file 을 stream 으로 만들어 다운 받는 방식 (서버에 파일이 저장 되지 않음)
# 두 번째는 static file 로 만들어서 그 파일을 다운로드 하는 방식

from flask import Flask, template_rendered

app = Flask(__name__)

@app.route('/')
def home():
    return 'this page is first page, please move to upload page'

@app.route('/file_upload')
def upload():
    return '''
    <a href='/wav_file_download_with_file'>Click me.</a>

    <form method='get' action='wav_file_download_with_file'>
        <button type = 'submit'>Download!</button>
    </form>
    '''

if __name__ == '__main__':
    app.run()