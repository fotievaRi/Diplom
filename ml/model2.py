import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf  # tensorflow
import glob
import re  # regular expression


def load_data2():
    train_data = []  # we will store the train data here
    test_data = []  # we will store the test data here
    sizes = []

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
            label = None  # get the training label from the dictionary created above
            # sample_data = pd.read_csv(filename, index_col=' ')  # finally read the accelerometer data
            sample_data_tmp = pd.read_csv(filename)  # finally read the accelerometer data
            sample_data = pd.DataFrame(sample_data_tmp, columns=['x', 'y', 'z'])
            # store the data in a dictionary for further processing later
            sample = {"sample_id": sample_id, "label": label, "sample_data": sample_data}
            test_data.append(sample)
            sizes.append(sample_data.shape[0])
            count_id += 1

    return train_data, test_data, sizes


def process_sample_data2(data, labels):
    X = []
    y = []
    original_sample_ids = []

    for d in data:
        original_sample_id = d['sample_id']
        label = d['label']
        sample_data = d['sample_data']
        # sample_data = pd.DataFrame(scaler.transform(sample_data), columns = d['sample_data'].columns)

        # Extracting features. In this example we collapse the time series into mean and std for the accelerometer values
        # features = sample_data[['userAcceleration.x', 'userAcceleration.y', 'userAcceleration.z']].agg(['mean','std']).unstack().tolist()
        features = sample_data[['x', 'y', 'z']].agg(
            ['mean', 'std', 'max', 'min']).unstack().tolist()
        padded = tf.keras.preprocessing.sequence.pad_sequences(sample_data.values.T, 12 * 1024, dtype=float).T
        X.append(padded)
        y.append(label)
        original_sample_ids.append(original_sample_id)

        # convert to X to numpy array (you can also directly store your features in numpy)
    X = np.array(X)

    # one-hot encode Y (expected by softmax classification)
    if labels is None:
        y = None
    else:
        y = pd.get_dummies(y)  # get dummies performs one-hot encode. We also use the
        y = np.array(y)

    return X, y, original_sample_ids


def create_keras_model2(num_classes):
    model = tf.keras.Sequential()
    # Adds a densely-connected layer with 64 units to the model:

    model.add(tf.keras.layers.Conv1D(filters=128, kernel_size=5))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=5, ))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=5, ))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
    model.add(tf.keras.layers.Dropout(0.1))
    model.add(tf.keras.layers.Conv1D(filters=32, kernel_size=3, ))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
    model.add(tf.keras.layers.Dropout(0.1))

    model.add(tf.keras.layers.Conv1D(filters=32, kernel_size=3, ))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
    model.add(tf.keras.layers.Dropout(0.1))

    model.add(tf.keras.layers.Conv1D(filters=16, kernel_size=3, ))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
    model.add(tf.keras.layers.Dropout(0.1))

    model.add(tf.keras.layers.Conv1D(filters=16, kernel_size=3, ))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
    model.add(tf.keras.layers.Dropout(0.1))
    model.add(tf.compat.v1.keras.layers.CuDNNLSTM(64))
    model.add(tf.keras.layers.LSTM(64))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(16, activation='relu'))
    # Add a softmax layer with num_classes output units:
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


# simple model training.
# you might want to avoid overfitting by monitoring validation loss and implement early stopping, etc
def train_model2(model, X, y):
    model.fit(X, y, epochs=100, batch_size=32, validation_split=0.0)


def predict2(model, X):
    y_pred = model.predict(X, batch_size=32)
    return y_pred


labels = ["dws", "ups", "sit", "std", "wlk", "jog"]

# load data
train_data, test_data, sizes = load_data2()


train_X, train_y, train_sample_ids = process_sample_data2(train_data, labels)
test_X, _, test_sample_ids = process_sample_data2(test_data, None)

# running it
model = create_keras_model2(6)
train_model2(model, train_X, train_y)
model.save("model2.h5")
model.summary()
# model_train = tf.keras.models.load_model("model2.h5")

y_pred = predict2(model, test_X)

# convert predictions to the kaggle format
y_pred_numerical = np.argmax(y_pred, axis=1)  # one-hot to numerical
y_pred_cat = [labels[x] for x in y_pred_numerical]  # numerical to string label

# generate the table with the correct IDs for kaggle.
# we get the correct sample ID from the stored array (test_sample_ids)
submission_results = pd.DataFrame({'id': test_sample_ids, 'label': y_pred_cat})
submission_results.to_csv("submission2.csv", index=False)
