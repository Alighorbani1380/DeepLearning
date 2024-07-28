# Sentiment Analysis Using IMDB Dataset  

This project implements a sentiment analysis model using a deep learning approach on the IMDB dataset. The objective is to classify movie reviews as positive or negative based on the text content.  

# Requirements

Ensure you have the following libraries installed:

Python (3.6 or newer)
NumPy
Matplotlib
TensorFlow (2.x)
Keras (included in TensorFlow)
You can install the required packages using pip:

pip install numpy matplotlib tensorflow  
# Dataset
The dataset used in this project is the IMDB dataset available through Keras. It contains 25,000 movie reviews for training and 25,000 for testing, categorized as positive or negative.

# Model Architecture
The model consists of a simple feedforward neural network with:

Input layer with 10,000 features (one-hot encoded)
Two hidden layers with 16 neurons each using ReLU activation
Output layer with 1 neuron using sigmoid activation for binary classification
#Training the Model
The model is trained over 20 epochs with a validation set to monitor the performance. Early stopping is employed to prevent overfitting.

Example code snippet for training  
history = model.fit(partial_x_train, partial_y_train,   
                    epochs=20,   
                    batch_size=512,   
                    validation_data=(x_val, y_val),   
                    callbacks=[early_stopping_monitor])  
# Results
After training, the model's performance is evaluated on the test set, providing insights into its accuracy and loss. The results are visualized through training and validation loss and accuracy graphs.
![image](https://github.com/user-attachments/assets/671d0762-7893-46c5-99df-bbd200761824)
![image](https://github.com/user-attachments/assets/761033bf-e28d-4f1c-b0ab-d3fd699ba5d3)

