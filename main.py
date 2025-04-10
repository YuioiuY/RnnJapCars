import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

df = pd.read_csv('japan_cars_dataset.csv') 
df = df.dropna()

target_col = 'price'
X = df.drop(columns=[target_col])
y = df[target_col]

numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)  
])

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.15, random_state=42)

model_pipeline = Pipeline([
    ('preprocessor', preprocessor)
])
X_train_processed = model_pipeline.fit_transform(X_train)
X_val_processed = model_pipeline.transform(X_val)
X_test_processed = model_pipeline.transform(X_test)

def mape_metric(y_true, y_pred):
    return tf.reduce_mean(tf.abs((y_true - y_pred) / tf.clip_by_value(tf.abs(y_true), 1e-8, tf.reduce_max(y_true)))) * 100

model = keras.Sequential([
    layers.Input(shape=(X_train_processed.shape[1],)),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
])

model.compile(optimizer='adam',
              loss='mse',
              metrics=['mae', mape_metric])  

history = model.fit(
    X_train_processed, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_val_processed, y_val),
    verbose=1
)

y_val_pred = model.predict(X_val_processed)
mse = mean_squared_error(y_val, y_val_pred)
mape = mean_absolute_percentage_error(y_val, y_val_pred)

print(f"\nMSE на валидации: {mse:.2f}")
print(f"Средняя процентная ошибка: {mape * 100:.2f}%")

y_test_pred = model.predict(X_test_processed)
test_mse = mean_squared_error(y_test, y_test_pred)
test_mape = mean_absolute_percentage_error(y_test, y_test_pred)
print(f"\nMSE на тестовой выборке: {test_mse:.2f}")
print(f"Процентная ошибка на тесте: {test_mape * 100:.2f}%")

plt.figure(figsize=(8, 5))
plt.scatter(y_val, y_val_pred, alpha=0.5)
plt.xlabel("Настоящая цена")
plt.ylabel("Предсказанная цена")
plt.title("Сравнение настоящей и предсказанной цены (валидация)")
plt.grid(True)
plt.show()
