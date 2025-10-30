import os
import pandas as pd
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from src.data_processing import load_data, preprocess_data
from src.analysis import perform_analysis
from src.model import create_model, compile_model, train_model, evaluate_model
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'data'
app.config['ALLOWED_EXTENSIONS'] = {'csv'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            df = load_data(file_path)
            columns = df.columns.tolist()
            return render_template('select_target.html', columns=columns, filename=filename)
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    filename = request.form['filename']
    target_column = request.form['target_column']
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    # Load data
    df = load_data(file_path)

    # Perform analysis
    stats = perform_analysis(df)

    # Get paths to the generated plots
    image_files = [f for f in os.listdir('static') if f.endswith('.png')]

    # Preprocess data
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df, target_column)

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Encode labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    num_classes = len(label_encoder.classes_)

    # Create and compile model
    input_shape = X_train_processed.shape[1]
    model = create_model(input_shape, num_classes)
    compile_model(model)

    # Train model
    history = train_model(model, X_train_processed, y_train_encoded)

    # Evaluate model
    loss, accuracy = evaluate_model(model, X_test_processed, y_test_encoded)

    return render_template('results.html', stats=stats.to_html(), images=image_files, loss=loss, accuracy=accuracy)

if __name__ == '__main__':
    app.run(debug=True)
