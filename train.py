import numpy as np
from sklearn.model_selection import train_test_split
from keras.layers import Conv2D, MaxPool2D, Dense, Dropout, Flatten
from keras.models import Sequential
from sklearn import metrics
import tensorflow as tf
from enum import Enum


class Train:
    @staticmethod
    def train_age_model(images, ages, filename):
        # Creating training and testing data
        x_train_age, x_test_age, y_train_age, y_test_age = train_test_split(images, ages, random_state=42)
        print(x_test_age[0])

        # Creating the network
        age_model = Sequential()
        age_model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(200, 200, 3)))  #128
        age_model.add(MaxPool2D(pool_size=3, strides=2))

        age_model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(200, 200, 3)))  #128
        age_model.add(MaxPool2D(pool_size=3, strides=2))

        age_model.add(Conv2D(128, kernel_size=3, activation='relu', input_shape=(200, 200, 3)))  #256
        age_model.add(MaxPool2D(pool_size=3, strides=2))

        age_model.add(Conv2D(256, kernel_size=3, activation='relu', input_shape=(200, 200, 3)))  #512
        age_model.add(MaxPool2D(pool_size=3, strides=2))

        age_model.add(Flatten())
        age_model.add(Dropout(0.2))
        age_model.add(Dense(256, activation='relu'))   #512
        age_model.add(Dense(1, activation='linear', name='age'))

        # Compile the model
        age_model.compile(optimizer='adam', loss='mse', metrics=['mae'])

        # Train the model
        history_age = age_model.fit(x_train_age, y_train_age, validation_data=(x_test_age, y_test_age), epochs=50,
                                    callbacks=[tf.keras.callbacks.EarlyStopping(
                                        monitor='val_loss',
                                        patience=5,
                                        restore_best_weights=True
                                    )])

        # Save the model to file
        age_model.save(filename)

        # Model accuracy
        predictions = age_model.predict(x_test_age)
        y_pred = (np.rint(predictions)).astype(int)[:, 0]
        print("Accuracy = ", metrics.accuracy_score(y_test_age, y_pred))  # Add range based metrics

    @staticmethod
    def train_gender_model(images, genders, filename):
        # Creating training and testing data
        x_train_gender, x_test_gender, y_train_gender, y_test_gender = train_test_split(images, genders, random_state=42)

        # Creating the network
        gender_model = Sequential()
        gender_model.add(Conv2D(36, kernel_size=3, activation='relu', input_shape=(200, 200, 3)))
        gender_model.add(MaxPool2D(pool_size=3, strides=2))

        gender_model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(200, 200, 3)))
        gender_model.add(MaxPool2D(pool_size=3, strides=2))

        gender_model.add(Conv2D(128, kernel_size=3, activation='relu', input_shape=(200, 200, 3)))
        gender_model.add(MaxPool2D(pool_size=3, strides=2))

        gender_model.add(Conv2D(256, kernel_size=3, activation='relu', input_shape=(200, 200, 3)))
        gender_model.add(MaxPool2D(pool_size=3, strides=2))

        gender_model.add(Conv2D(512, kernel_size=3, activation='relu', input_shape=(200, 200, 3)))
        gender_model.add(MaxPool2D(pool_size=3, strides=2))

        gender_model.add(Flatten())
        gender_model.add(Dropout(0.2))
        gender_model.add(Dense(512, activation='relu'))
        gender_model.add(Dense(1, activation='sigmoid', name='gender'))

        # Compile the model
        gender_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Train the model
        history_gender = gender_model.fit(x_train_gender, y_train_gender, validation_data=(x_test_gender, y_test_gender),
                                       epochs=20,
                                       callbacks=[tf.keras.callbacks.EarlyStopping(
                                           monitor='val_loss',
                                           patience=5,
                                           restore_best_weights=True
                                       )])

        # Save the model to file
        gender_model.save(filename)

        # Model accuracy
        predictions = gender_model.predict(x_test_gender)
        y_pred = (predictions >= 0.5).astype(int)[:, 0]
        print("Accuracy = ", metrics.accuracy_score(y_test_gender, y_pred))

    @staticmethod
    def train_race_model(images, races, filename):
        # Creating training and testing data
        x_train_race, x_test_race, y_train_race, y_test_race = train_test_split(images, races, random_state=42)

        shape = images[0].shape
        # Creating the network
        race_model = Sequential()
        race_model.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=(200, 200, 3)))
        race_model.add(MaxPool2D(pool_size=3, strides=2))

        race_model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(200, 200, 3)))
        race_model.add(MaxPool2D(pool_size=3, strides=2))

        race_model.add(Conv2D(128, kernel_size=3, activation='relu', input_shape=(200, 200, 3)))
        race_model.add(MaxPool2D(pool_size=3, strides=2))

        race_model.add(Conv2D(256, kernel_size=3, activation='relu', input_shape=(200, 200, 3)))
        race_model.add(MaxPool2D(pool_size=3, strides=2))

        race_model.add(Conv2D(512, kernel_size=3, activation='relu', input_shape=(200, 200, 3)))
        race_model.add(MaxPool2D(pool_size=3, strides=2))

        race_model.add(Flatten())
        race_model.add(Dropout(0.2))
        race_model.add(Dense(512, activation='relu'))
        race_model.add(Dense(5, activation='softmax', name='race'))

        # Compile the model
        race_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']) # TODO pobawić się tym

        # Train the model
        history_gender = race_model.fit(x_train_race, y_train_race,
                                          validation_data=(x_test_race, y_test_race),
                                          epochs=50,
                                          callbacks=[tf.keras.callbacks.EarlyStopping(
                                              monitor='val_loss',
                                              patience=5,
                                              restore_best_weights=True
                                          )])

        # Save the model to file
        race_model.save(filename)

        # Model accuracy
        score = race_model.evaluate(x_test_race, y_test_race, verbose=0)
        print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')
        # predictions = race_model.predict(x_test_race)
        # y_pred = (np.rint(predictions)).astype(int)[:, 0]
        # print("Accuracy = ", metrics.accuracy_score(y_test_race, y_pred))


class Gender(Enum):
    Male = 0
    Female = 1


class Race(Enum):
    White = 0
    Black = 1
    Asian = 2
    Indian = 3
    Other = 4
