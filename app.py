from flask import Flask, render_template, request, redirect, url_for
from predict_iris import predict_iris
import os

app = Flask(__name__)

# Configurer le dossier pour les fichiers uploadés
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/authentification', methods=['GET', 'POST'])
def authentification():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        result = predict_iris(filepath)
        return render_template('result.html', result=result)
    return render_template('authentification.html')

if __name__ == '__main__':
    app.run(debug=True)