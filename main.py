import numpy as np
import tensorflow as tf
from keras.src.models import Sequential
from keras.src.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.src.metrics import Precision, Recall, BinaryAccuracy
from keras.src.saving import saving_api
import pickle
import os


# Initialize dataset
data = tf.keras.utils.image_dataset_from_directory('data_training', batch_size=10, image_size=(512, 512), shuffle=True)

# Scale the data from 255.0 to 1.0 for optimization
data = data.map(lambda x, y: (x / 255, y))

# Happy = Label 0
# Sad = Label 1

# Iterate the data as a batch throughout the dataset
batch = data.as_numpy_iterator().next()

# Get the size of each type of data
train_size = int(len(data) * 0.7)
valid_size = int(len(data) * 0.2) + 1
test_size = int(len(data) * 0.1) + 1

# Split the data into 3 categories: Traing, Validation, Testing
train_data = data.take(train_size)
valid_data = data.skip(train_size).take(valid_size)
test_data = data.skip(train_size + valid_size).take(test_size)

def build():
    # Building the model
    model = Sequential()
    model.add(Conv2D(16, (3, 3), 1, activation='relu', input_shape=(512, 512, 3)))
    model.add(MaxPooling2D())
    model.add(Dropout(0.25))

    model.add(Conv2D(32, (3, 3), 1, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.25))

    model.add(Conv2D(16, (3, 3), 1, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    return model

def compile(model):
    # Compile the model
    model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])
    model.summary()

def train(model):
    # Log the performance
    logs = tf.keras.callbacks.TensorBoard(log_dir='logs')
    info = model.fit(train_data, epochs=30, validation_data=valid_data, callbacks=[logs])
    return info, logs

# Build and train the model
modell = build()
compile(modell)
info, logs = train(modell)

# Setting up for testing data
pre = Precision()
re = Recall()
acc = BinaryAccuracy()

# Test using test_data
for batch in test_data.as_numpy_iterator():
    x, y = batch
    yhat = modell.predict(x)
    pre.update_state(y, yhat)
    re.update_state(y,yhat)
    acc.update_state(y, yhat)
print(f'Precision: {pre.result().numpy()}, Recall: {re.result().numpy()}, Accuracy: {acc.result().numpy()}')

# Save model for later uses
saving_api.save_model(modell ,os.path.join('models', 'emote_model.keras'))

# Saving training data for visualization
with open('training_history.pkl', 'wb') as f:
    pickle.dump(info, f)
