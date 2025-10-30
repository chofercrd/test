import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

def create_model(input_shape, num_classes):
    """Creates a 4-layer fully connected neural network."""
    model = Sequential([
        Input(shape=(input_shape,)),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    return model

def compile_model(model):
    """Compiles the model."""
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

def train_model(model, X_train, y_train, epochs=10, batch_size=32):
    """Trains the model."""
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)
    return history

def evaluate_model(model, X_test, y_test):
    """Evaluates the model."""
    loss, accuracy = model.evaluate(X_test, y_test)
    return loss, accuracy
