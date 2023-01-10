import numpy as np
from sklearn.model_selection import train_test_split
from keras.layers import Conv2D, MaxPool2D, Dense, Dropout, Flatten
from keras.models import Sequential
from sklearn import metrics
import tensorflow as tf
from enum import Enum
import matplotlib.pyplot as plt

class Train:
    @staticmethod
    def train_age_model(images, ages, filename):
        # Creating training and testing data 1632
        x_train_age, x_test_age, y_train_age, y_test_age = train_test_split(images, ages,
                                                                            random_state=42, stratify=ages)
        # print(len(y_train_age), len(y_test_age))
        plt.figure(1)
        plt.hist(y_train_age, bins=100)
        plt.figure(2)
        plt.hist(y_test_age, bins=100)
        plt.show()

        age_intervals = ['(1,5)', '(6,10)', '(11,15)', '(16,20)', '(21,25)', '(26,30)', '(31,35)', '(36,40)',
                         '(41,45)', '(46,50)', '(51,55)', '(56,60)', '(61,65)', '(66,70)', '(71,75)', '(76,80)',
                         '(81,85)', '(86,90)', '(91,95)', '(96,100)']

        # Creating the network
        age_model = Sequential()
        age_model.add(Conv2D(128, kernel_size=3, activation='relu', input_shape=(200, 200, 3)))
        age_model.add(MaxPool2D(pool_size=3, strides=2))

        age_model.add(Conv2D(128, kernel_size=3, activation='relu', input_shape=(200, 200, 3)))
        age_model.add(MaxPool2D(pool_size=3, strides=2))

        age_model.add(Conv2D(256, kernel_size=3, activation='relu', input_shape=(200, 200, 3)))
        age_model.add(MaxPool2D(pool_size=3, strides=2))

        age_model.add(Conv2D(512, kernel_size=3, activation='relu', input_shape=(200, 200, 3)))
        age_model.add(MaxPool2D(pool_size=3, strides=2))

        age_model.add(Flatten())
        age_model.add(Dropout(0.2))
        age_model.add(Dense(512, activation='relu'))
        age_model.add(Dense(1, activation='softmax', name='age'))

        # Compile the model
        age_model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy, metrics=['mae'])

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

        # Old solution

        # predictions = age_model.predict(x_test_age)
        # y_pred = (np.rint(predictions)).astype(int)[:, 0]
        #
        # def accuracy(y_pred, y_test_age):
        #     acc = [1 for pred, age in zip(y_pred, y_test_age) if pred in range(age - 5, age + 5)]
        #
        #     return sum(acc) / len(y_pred)
        #
        # print("Accuracy = ", accuracy(y_pred, y_test_age))

        score = age_model.evaluate(x_test_age, y_test_age, verbose=0)
        print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')

    @staticmethod
    def train_gender_model(images, genders, filename):
        # Creating training and testing data
        x_train_gender, x_test_gender, y_train_gender, y_test_gender = train_test_split(images, genders,
                                                                                        random_state=42,
                                                                                        stratify=genders)

        plt.figure(1)
        plt.hist(y_train_gender, bins=2)
        plt.figure(2)
        plt.hist(y_test_gender, bins=2)
        plt.show()

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
                                       epochs=50,
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
        x_train_race, x_test_race, y_train_race, y_test_race = train_test_split(images, races,
                                                                                random_state=42, stratify=races)

        plt.figure(1)
        plt.hist(y_train_race, bins=5)
        plt.figure(2)
        plt.hist(y_test_race, bins=5)
        plt.show()

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
        race_model.add(Dense(5, activation='softmax', name='race')) #softmax też do wieku

        # Compile the model
        race_model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy, metrics=['accuracy']) # loss żeby był %

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

        # loss_function = tf.keras.losses.SparseCategoricalCrossentropy()
        #
        # predictions = race_model.predict(x_test_race)
        # loss = loss_function(y_test_race, predictions)
        # print('Loss = ', loss)
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
