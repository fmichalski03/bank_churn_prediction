import numpy as np
import tensorflow as tf
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

ann.fit(X_train, y_train, batch_size=16, epochs=100, callbacks=[early_stopping])

y_pred = ann.predict(X_test)
y_pred = (y_pred>0.5)

wrong = 0
for i in range(len(y_test)):
    if y_pred[i] != y_test[i]:
        wrong  += 1

print(1 - wrong/len(y_test))
print(len(y_pred))


        


