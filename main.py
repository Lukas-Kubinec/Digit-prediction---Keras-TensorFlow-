"""
KNOWLEDGE REPRESENTATION AND REASONING CW2
Description: Image recognition
Author: Lukas Kubinec
"""

# Importing NumPy library used for its array functionalities
import numpy as np
# Importing of Systems routines to control various parameters
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] =   '1' # Enables the OneDNN optimization
os.environ['TF_CPP_MIN_LOG_LEVEL'] =    '1' # Disables INFO debugging messages (removes rounding error warning)
# Importing of library used for plotting images
import matplotlib.pyplot as plt
# Importing a dataset containing 60,000 training / 10,000 testing handwritten digits from the Keras library
from keras.src.datasets import mnist
# Import of the training model and its functionalities
from keras import Sequential
from keras.src.layers import Dense, Flatten
from keras.src.utils import to_categorical

# --- Initial variables ---
colour_scheme = "Blues_r" # Colour scheme used for data plotting
is_running = True # Used to run the main program loop
training_model_validation_split = 0.2 # 0.2 = 20% Fraction of the training data to be used as validation data
training_epochs_amount = 10 # Controls how many epochs are run when training model
testing_amount = 200 # Controls how many images are used for number prediction

# Loads the Mnist dataset into separate training and testing variables - https://keras.io/api/datasets/mnist/
(train_x, train_y), (test_x, test_y) = mnist.load_data()

# --- KERAS SEQUENTIAL MODEL ---
# Creation of the Sequential model
model = Sequential([
    Flatten(input_shape=(28, 28)),  # Flattens the 28x28 images
    Dense(128, activation='relu'),  # Hidden layer with RELU activation
    Dense(10, activation='softmax') # Output layer with SOFTMAX activation
])
# Compilation of the Keras Sequential model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# --- METHODS ---
# Normalises input dataset and converts dataset labels to 'one-hot' encoding
def dataset_normalisation():
    # Normalisation of the input dataset
    _train_x = train_x / 255.0
    _test_x = test_x / 255.0
    # Converts dataset labels to one-hot encoding
    _train_y = to_categorical(train_y, 10)
    _test_y = to_categorical(test_y, 10)
    # Verifying shapes
    print("Shape of train_y:", _train_y.shape)  # Should output (60000, 10)
    print("Shape of test_y:", _test_y.shape)  # Should output (10000, 10)
    return _train_x, _test_x, _train_y, _test_y # Returns the updated values

# Method that runs the model training
def train_model(_epochs, _validation_split):
    # Training the model
    print("Training of the model:")
    model.fit(train_x, train_y, verbose=1, epochs=_epochs, batch_size=32, validation_split=_validation_split)
    # Model evaluation
    print("Evaluation of the model:")
    data_loss, data_accuracy = model.evaluate(test_x, test_y, verbose=1)
    print("Data loss: {:.3f} | Data accuracy: {:.3f}".format(data_loss, data_accuracy))

# Plotting of digits in the dataset
def plot_training_dataset(_set_x, _set_y, _amount):
    # Explore digits in the training dataset
    for i, image in enumerate(_set_x[0:_amount]):
        # Prints information about the current dataset
        print("Image number: {} | Actual number: {}".format(i, _set_y[i]))
        # Sets the left/right title of the plot
        plt.title("Image n. {}".format(i), loc="left")          # Writes the numer of the current image
        plt.title("Digit {}".format(_set_y[i]), loc="right")    # Writes the actual digit value
        # Disables the axis labels
        plt.axis("off")
        # Draws/Plots the current dataset image
        plt.imshow(image, cmap=colour_scheme)
        plt.show()

# Trained Model predicting from dataset
def model_predict_from_dataset(_train_x, _train_y, _amount):
    success_rate = 0
    # Predicts a number from the dataset
    for i in range(_amount):
        print("\n[{}] Predicting now...".format(i))
        # Select a single image from the test set (e.g., the first image)
        single_image = _train_x[i]  # Assuming test_x contains normalized test data
        single_image = np.expand_dims(single_image, axis=0)  # Add batch dimension
        # Predict the class probabilities
        predicted_probabilities = model.predict(single_image, verbose=1) # Generates output predictions
        predicted_number = np.argmax(predicted_probabilities) # Returns the highest value
        actual_number = np.argmax(_train_y[i])  # Returns the highest value
        # Print of final result
        print("Predicted number: {} | Actual number: {}".format(predicted_number, actual_number))
        # Check if the prediction is the same as the actual number
        if predicted_number == actual_number:
            print("Predicted number is correct!")
            success_rate += 1
        else:
            print("Predicted number is incorrect!")
    success_rate = success_rate / _amount * 100
    print("\nSuccess prediction rate: {:.2f}%".format(success_rate))

# --- MAIN LOOP ---
while is_running:
    print("CW2 - Image recognition")
    # Plot of the training dataset
    print("\nTraining dataset")
    plot_training_dataset(train_x, train_y, 5)

    # Plot of the testing dataset
    print("\nTesting dataset")
    plot_training_dataset(test_x, test_y, 5)

    # Normalises input dataset and converts dataset labels to 'one-hot' encoding, so it can be used to train the model
    print("\nNormalising and Converting the dataset")
    train_x, test_x, train_y, test_y = dataset_normalisation()

    # Running the training of the model
    print("\nTraining model on dataset")
    train_model(training_epochs_amount, training_model_validation_split) # Trains the recognition model

    # Runs the model to predict the digits from the dataset
    print("\nPredicting numbers from dataset")
    model_predict_from_dataset(test_x, test_y, testing_amount)

    # Ends the program
    print("\nGoodbye!")
    is_running = False