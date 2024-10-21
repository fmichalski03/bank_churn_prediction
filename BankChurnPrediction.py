import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dropout
import pandas as pd
from sklearn.compose import  ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

dataset = pd.read_csv("bank_customer_churn_prediction.csv")
X = dataset.iloc[:,1:-1].values
y = dataset.iloc[:, -1].values

print(y)

# Encoding
ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[1, 2])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# Feature scaling

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Creating  the ANN model

ann = tf.keras.models.Sequential()

#  Adding the layers




ann.add(tf.keras.layers.Dense(units=64, activation='relu'))  # Druga warstwa ukryta 32 neurony
ann.add(Dropout(0.20))
ann.add(tf.keras.layers.Dense(units=32, activation='relu'))  # Druga warstwa ukryta 32 neurony
ann.add(Dropout(0.20))
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))  # Warstwa wyjściowa





ann.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9), loss='binary_crossentropy', metrics=['accuracy'])

# Training the ANN model

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = ann.fit(X_train, y_train, batch_size=16, epochs=100, validation_data=(X_test, y_test))

y_pred = ann.predict(X_test)
y_pred = (y_pred>0.5)

wrong = 0
for i in range(len(y_test)):
    if y_pred[i] != y_test[i]:
        wrong  += 1

print(1 - wrong/len(y_test))
print(len(y_pred))

plt.rc('font', size=12)
plt.rc('axes', titlesize=16) 
plt.rc('axes', labelsize=14)
plt.rc('legend', fontsize=12)


plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Train Loss', color='blue', linestyle='--', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss', color='red', linestyle='-', linewidth=2)
plt.title('Train and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))  # Zmieniamy rozmiar wykresu
plt.plot(history.history['accuracy'], label='Train Accuracy', color='green', linestyle='--', linewidth=2)
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange', linestyle='-', linewidth=2)
plt.title('Train and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()


        


