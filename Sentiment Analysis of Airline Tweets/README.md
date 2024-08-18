# Sentiment Analysis of Airline Tweets  

This project aims to perform sentiment analysis on tweets related to airlines using two machine learning models: Logistic Regression and a Neural Network. The analysis helps identify the sentiment expressed in tweets as positive, negative, or neutral.  

## Steps Involved  

1. **Data Loading**  
2. **Data Preprocessing**  
3. **Splitting Data into Training and Testing Sets**  
4. **Implementing the Logistic Regression Model**  
5. **Implementing the Neural Network Model**  
6. **Training and Evaluating the Models**  
7. **Calculating Accuracy Metrics for Both Models**  

### Libraries Used  
- For data manipulation and loading: `pandas`  
- For text vectorization: `scikit-learn`  
- For machine learning models: `scikit-learn`, `TensorFlow`, or `Keras`  

---  

## Data Loading and Preprocessing:  
```python  
import pandas as pd  
from sklearn.model_selection import train_test_split  
from sklearn.feature_extraction.text import CountVectorizer  

data = pd.read_csv('Tweets.csv')  
X = data['text']  
L = data['airline_sentiment']  
y = []  
for s in L:  
    if s == 'negative':  
        y.append(0)  
    elif s == 'positive':  
        y.append(1)  
    else:  
        y.append(2)  

vectorizer = CountVectorizer()  
X = vectorizer.fit_transform(X)
```
## Split Data into Training and Testing Sets:
```python  
from sklearn.model_selection import train_test_split  
import numpy as np  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  
y_train = np.asarray(y_train).astype("float32")  
y_test = np.asarray(y_test).astype("float32")
```
## Implementing the Logistic Regression Model:
```python
from sklearn.linear_model import LogisticRegression  
from sklearn.metrics import accuracy_score  

LogisticRegression_model = LogisticRegression(max_iter=1000)  
LogisticRegression_model.fit(X_train, y_train)  
LogisticRegression_pred = LogisticRegression_model.predict(X_test)  
LogisticRegression_accuracy = accuracy_score(y_test, LogisticRegression_pred)  

print("Logistic Regression accuracy:", LogisticRegression_accuracy)
```
## Implementing the Neural Network Model:
```python
from tensorflow.keras.models import Sequential  
from tensorflow.keras.layers import Dense  

model = Sequential([  
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),  
    Dense(3, activation='softmax')  
])  
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])  
model.fit(X_train.toarray(), y_train, epochs=3, batch_size=32)  
nn_accuracy = model.evaluate(X_test.toarray(), y_test)[1]  
print("Neural Network accuracy:", nn_accuracy)
```
Outputs:
```
Logistic Regression accuracy: 0.798155737704918  
Epoch 1/3  
366/366 [==============================] - 5s 12ms/step - loss: 0.6069 - accuracy: 0.7561  
Epoch 2/3  
366/366 [==============================] - 4s 10ms/step - loss: 0.3464 - accuracy: 0.8710  
Epoch 3/3  
366/366 [==============================] - 4s 11ms/step - loss: 0.2234 - accuracy: 0.9238  
92/92 [==============================] - 0s 4ms/step - loss: 0.5419 - accuracy: 0.8046  
Neural Network accuracy: 0.8046448230743408
```
