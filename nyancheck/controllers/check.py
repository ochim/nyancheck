import os
import sqlite3
from flask import Blueprint, render_template, request, redirect, url_for, send_from_directory, session
from werkzeug import secure_filename
from nyancheck.net.predict import predict

app = Blueprint('check', __name__, template_folder='templates', static_folder="./static", static_url_path="/static")

upload_dir = './uploads'
allowed_extensions = set(['png', 'jpg', 'gif'])
config = {}
config['upload_dir'] = upload_dir

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1] in allowed_extensions

@app.route("/")
def index():
    return render_template('index.html')

@app.route('/<path:path>')
def static_file(path):
    return app.send_static_file(path)

@app.route('/api/v1/send', methods=['GET', 'POST'])
def send():
    if request.method == 'POST':
        img_file = request.files['img_file']
        if img_file and allowed_file(img_file.filename):
            filename = secure_filename(img_file.filename)
            img_file.save(os.path.join(config['upload_dir'], filename))
            img_url = '/uploads/' + filename
            nyan_type = predict(filename)
            return render_template('index.html', img_url=img_url, nyan_type=nyan_type)
        else:
            return ''' <p>許可されていない拡張子です</p> '''
    else:
        return redirect(url_for(''))

@app.route('/api/v1/check/<filename>', methods=['GET', 'POST'])
def check(filename):
    if request.method == 'POST':
        img_url = '/uploads/' + filename
        nyan_type = predict(filename)
        return render_template('index.html', img_url=img_url, nyan_type=nyan_type)
    else:
        return redirect(url_for(''))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(config['upload_dir'], filename)

if __name__ == '__main__':
    app.debug = True
    app.run()
