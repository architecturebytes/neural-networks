import numpy as np
import tensorflow as tf
from tensorflow import keras

# Features: Age, BMI, Blood Pressure, Cholesterol Level, Exercise Hours
X = np.array([
    [35, 25, 120, 180, 3],
    [45, 28, 135, 210, 2],
    [28, 24, 110, 160, 1],
    [60, 30, 140, 240, 0],
    [50, 29, 130, 200, 2],
    [40, 27, 125, 190, 2]
])

y = np.array([0, 1, 0, 1, 1, 0])

# Build a simple feedforward neural network
model = keras.Sequential()

# Input layer with 5 features
model.add(keras.layers.Input(shape=(5,)))

# Hidden layer with 8 neurons and ReLU activation
model.add(keras.layers.Dense(8, activation='relu'))

# Another hidden layer with 4 neurons and ReLU activation
model.add(keras.layers.Dense(4, activation='relu'))

# Output layer with 1 neuron and sigmoid activation 
model.add(keras.layers.Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

# Train the model
model.fit(X, y, epochs=1000, verbose=1)

# Let's define some data for prediction 
new_data = np.array([
    [30, 26, 125, 190, 2],
    [55, 31, 138, 220, 1]
])

# Make predictions on the new data
the_predictions = model.predict(new_data)

print("Predictions:", the_predictions)

rounded_predictions = [int(round(pred[0])) for pred in the_predictions]

print("Predicted Targets:", rounded_predictions)
