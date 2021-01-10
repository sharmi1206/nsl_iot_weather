import tensorflow as tf
import neural_structured_learning as nsl
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


df = pd.read_csv("../data/datatest2.csv", sep=",")

print(df.shape)
print(df.columns)

df = df.drop(columns = ['date'])

df['Temperature'] = df['Temperature'].astype(np.float32)

df['Humidity'] = df['Humidity'].astype(np.float32)
df['Light'] = df['Light'].astype(np.float32)
df['CO2'] = df['CO2'].astype(np.float32)
df['HumidityRatio'] = df['HumidityRatio'].astype(np.float32)


df_train_X = df[['Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio']]

df_train_y= df[['Occupancy']]

print(df.head())
df_x_train, df_x_test, df_y_train, df_y_test = train_test_split(df_train_X, df_train_y, test_size=0.3, random_state=42)

x_train = df_x_train.values
x_test = df_x_test.values


y_train = np.asarray(df_y_train.values.reshape((-1,))).astype(np.float32)
y_test = np.asarray(df_y_test.values.reshape((-1,))).astype(np.float32)

# Prepare data.
print(np.shape(x_train), np.shape(y_train), np.shape(x_test), np.shape(y_test), np.unique(y_train), np.unique(y_test))

# # Create a base model -- sequential, functional, or subclass.
model = tf.keras.Sequential([
    tf.keras.Input((x_train.shape[1]), name='feature'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(2, activation=tf.nn.softmax)
])

# Wrap the model with adversarial regularization.
adv_config = nsl.configs.make_adv_reg_config(multiplier=0.1, adv_step_size=0.01)
adv_model = nsl.keras.AdversarialRegularization(model, adv_config=adv_config, base_with_labels_in_features=True)

# Compile, train, and evaluate.
adv_model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
adv_model.fit({'feature': x_train, 'label': y_train}, batch_size=32, epochs=5)
prediction = adv_model.predict({'feature': x_test, 'label': y_test})
results = adv_model.evaluate({'feature': x_test, 'label': y_test}, return_dict=True)

print("Prediction", adv_model.predict({'feature': x_test, 'label': y_test}))
print(results)
