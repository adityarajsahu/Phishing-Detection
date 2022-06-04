import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import pandas as pd
from sklearn.model_selection import train_test_split

DataFrame = pd.read_csv('Dataset/Phising_Training_Dataset.csv')
# print(DataFrame.columns)

features = DataFrame.drop(columns=['key', 'Result'])
label = DataFrame['Result']

X_train, X_val, Y_train, Y_val = train_test_split(features, label, test_size=0.2, shuffle=True)

model = Sequential()
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(2, activation='relu'))
model.add(Dense(1, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.fit(X_train, Y_train, batch_size=8, epochs=10)
test_loss, test_acc = model.evaluate(X_val, Y_val)

print("Validation Accuracy: {:.2f} %".format(test_acc * 100))

