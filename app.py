import os
import pandas as pd
import joblib
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template, redirect, url_for, session
from werkzeug.utils import secure_filename
from src.data_processing import load_data, preprocess_data
from src.analysis import perform_analysis
from src.model import create_model, compile_model, train_model, evaluate_model
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)
app.secret_key = 'supersecretkey'  # Needed for Flask sessions
app.config['UPLOAD_FOLDER'] = 'data'
app.config['SAVED_OBJECTS_FOLDER'] = 'saved_objects'
app.config['ALLOWED_EXTENSIONS'] = {'csv'}

# Create directories if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['SAVED_OBJECTS_FOLDER'], exist_ok=True)


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

    # Clear old plots
    for f in os.listdir('static'):
        if f.endswith('.png'):
            os.remove(os.path.join('static', f))

    # Load data
    df = load_data(file_path)

    # Get feature names for the prediction form
    features = df.drop(columns=[target_column]).columns.tolist()

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

    # Save the model, preprocessor, and label encoder
    file_basename = filename.split('.')[0]
    model_path = os.path.join(app.config['SAVED_OBJECTS_FOLDER'], f'{file_basename}_model.keras')
    preprocessor_path = os.path.join(app.config['SAVED_OBJECTS_FOLDER'], f'{file_basename}_preprocessor.joblib')
    label_encoder_path = os.path.join(app.config['SAVED_OBJECTS_FOLDER'], f'{file_basename}_label_encoder.joblib')

    model.save(model_path)
    joblib.dump(preprocessor, preprocessor_path)
    joblib.dump(label_encoder, label_encoder_path)

    # Store results in session
    session['results'] = {
        'stats': stats.to_html(),
        'images': image_files,
        'loss': loss,
        'accuracy': accuracy,
        'filename': filename,
        'features': features
    }

    # Clean up the session to avoid memory leaks
    tf.keras.backend.clear_session()

    return render_template('results.html', **session['results'], prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    results = session.get('results', {})
    filename = results.get('filename')
    if not filename:
        return redirect(url_for('index'))

    file_basename = filename.split('.')[0]

    # Load the saved objects
    model_path = os.path.join(app.config['SAVED_OBJECTS_FOLDER'], f'{file_basename}_model.keras')
    preprocessor_path = os.path.join(app.config['SAVED_OBJECTS_FOLDER'], f'{file_basename}_preprocessor.joblib')
    label_encoder_path = os.path.join(app.config['SAVED_OBJECTS_FOLDER'], f'{file_basename}_label_encoder.joblib')

    model = tf.keras.models.load_model(model_path)
    preprocessor = joblib.load(preprocessor_path)
    label_encoder = joblib.load(label_encoder_path)

    features = results.get('features', [])

    # Get user input
    user_input = {feature: request.form[feature] for feature in features}

    # Create a DataFrame from the user input
    input_df = pd.DataFrame([user_input])

    # Process the input
    input_processed = preprocessor.transform(input_df)

    # Make a prediction
    prediction_proba = model.predict(input_processed)
    prediction_encoded = np.argmax(prediction_proba, axis=1)
    prediction = label_encoder.inverse_transform(prediction_encoded)[0]

    # Clean up the session
    tf.keras.backend.clear_session()

    return render_template('results.html', **results, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
