import numpy as np
import pandas as pd
import tensorflow as tf
import glob
from tensorflow import keras
from matplotlib import pylab
import re


def load_data():
    path_train = ["data/train/dws_1/sub_*.csv",
                  "data/train/dws_2/sub_*.csv",
                  "data/train/dws_11/sub_*.csv",
                  "data/train/jog_9/sub_*.csv",
                  "data/train/jog_16/sub_*.csv",
                  "data/train/sit_5/sub_*.csv",
                  "data/train/sit_13/sub_*.csv",
                  "data/train/std_6/sub_*.csv",
                  "data/train/std_14/sub_*.csv",
                  "data/train/ups_3/sub_*.csv",
                  "data/train/ups_4/sub_*.csv",
                  "data/train/ups_12/sub_*.csv",
                  "data/train/wlk_7/sub_*.csv",
                  "data/train/wlk_8/sub_*.csv",
                  "data/train/wlk_15/sub_*.csv"]

    path_test = ["data/test/dws_11/sub_*.csv",
                 "data/test/jog_16/sub_*.csv",
                 "data/test/sit_13/sub_*.csv",
                 "data/test/std_14/sub_*.csv",
                 "data/test/ups_12/sub_*.csv",
                 "data/test/wlk_15/sub_*.csv"]
    train_data = []  # we will store the train data here
    test_data = []  # we will store the test data here

    # read training labels
    labels = pd.read_csv("data/train_labels.csv", header=None)
    labels_values = labels.values
    label_dictionary = {a: labels_values[a - 1][0] for a in range(1, len(labels) + 1)}

    count_id = 1
    for path in path_train:
        for filename in glob.glob(path):
            sample_id = int(count_id)  # get the group from the filename (regular expression)
            label = label_dictionary[sample_id]  # get the training label from the dictionary created above
            sample_data_tmp = pd.read_csv(filename)  # finally read the accelerometer data
            sample_data = pd.DataFrame(sample_data_tmp, columns=['x', 'y', 'z'])
            # store the data in a dictionary for further processing later
            sample = {"sample_id": sample_id, "label": label, "sample_data": sample_data}
            train_data.append(sample)
            count_id += 1

    count_id = 1
    for path in path_test:
        for filename in glob.glob(path):
            sample_id = int(count_id)  # get the group from the filename (regular expression)
            label = None # get the training label from the dictionary created above
            # sample_data = pd.read_csv(filename, index_col=' ')  # finally read the accelerometer data
            sample_data_tmp = pd.read_csv(filename)  # finally read the accelerometer data
            sample_data = pd.DataFrame(sample_data_tmp, columns=['x', 'y', 'z'])
            # store the data in a dictionary for further processing later
            sample = {"sample_id": sample_id, "label": label, "sample_data": sample_data}
            test_data.append(sample)
            count_id += 1

    return train_data, test_data


def process_sample_data(data, labels):
    X = []
    y = []
    original_sample_ids = []

    for d in data:
        original_sample_id = d['sample_id']
        label = d['label']
        sample_data = d['sample_data']

        # Extracting features. In this example we collapse the time series into mean and std for the accelerometer values
        features = sample_data[['x', 'y', 'z']].agg(
            ['mean', 'std', 'max', 'min']).unstack().tolist()
        X.append(features)
        y.append(label)
        original_sample_ids.append(original_sample_id)

    # convert to X to numpy array (you can also directly store your features in numpy)
    X = np.array(X)

    # one-hot encode Y (expected by softmax classification)
    if labels is None:
        y = None
    else:
        y = pd.get_dummies(y)
        # y = pd.get_dummies(y)[labels]  # get dummies performs one-hot encode. We also use the "labels" list to make sure the order is as expected
        y = np.array(y)

    return X, y, original_sample_ids


def create_keras_model(num_classes):
    model = tf.keras.Sequential()
    # Adds a densely-connected layer with 64 units to the model:
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    # Add one more layer
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    # Add a softmax layer with num_classes output units:
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

    optimizer = tf.keras.optimizers.Adam()
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


# simple model training.
# you might want to avoid overfitting by monitoring validation loss and implement early stopping, etc
def train_model(model, X, y):
    model.fit(X, y, epochs=200, batch_size=32)


def predict(model, X):
    y_pred = model.predict(X, batch_size=32)
    return y_pred


labels = ["dws", "ups", "sit", "std", "wlk", "jog"]

# load data
train_data, test_data = load_data()
sample = train_data[20]
print(sample['label'])
print(sample['sample_id'])
# display(sample['sample_data'].head(3))
# x = sample['sample_data'].values[:, 0]
# y = sample['sample_data'].values[:, 1]
# z = sample['sample_data'].values[:, 2]


train_X, train_y, train_sample_ids = process_sample_data(train_data, labels)
test_X, _, test_sample_ids = process_sample_data(test_data, None)

print(train_X.shape)  # in this example we have 255 sample to train with 6 features each
print(train_y.shape)  # we have 6 possible labels for each sample ("dws","ups", "wlk", "jog", "std", "sit")

print(test_X.shape)  # we have 105 test samples. We don't have the labels for those. our model has to guess them from the input data

# running it
model = create_keras_model(6)

train_model(model, train_X, train_y)

model.save("model.h5")
model_train = keras.models.load_model("model.h5")

y_pred = predict(model_train, test_X)

# convert predictions to the kaggle format
y_pred_numerical = np.argmax(y_pred, axis=1)  # one-hot to numerical
y_pred_cat = [labels[x] for x in y_pred_numerical]  # numerical to string label

# generate the table with the correct IDs for kaggle.
# we get the correct sample ID from the stored array (test_sample_ids)
submission_results = pd.DataFrame({'id': test_sample_ids, 'label': y_pred_cat})
submission_results.to_csv("submission.csv", index=False)
