import cv2
import numpy as np
import matplotlib.pyplot as plt
from data import FaceData
from train import Train, Gender, Race
from utils import Utils
from keras.models import load_model
from random import randrange
import time
import tensorflow as tf
from sklearn import metrics
from sklearn.model_selection import train_test_split
import os


def main():
    # Initial operations
    data = FaceData()
    data.load_images()
    train = Train()
    utils = Utils()
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # Data
    images = np.array(data.images)
    ages = np.array(data.ages, dtype=np.int32)
    genders = np.array(data.genders, dtype=np.int32)
    races = np.array(data.races, dtype=np.int32)

    # Training models
    # train.train_age_model(images, ages, 'age_modeer.h5')
    # train.train_gender_model(images, genders, 'gender_model_15_55_filter.h5')
    # train.train_race_model(images, races, 'race_model_55_120.h5')

    # Load model
    gender_model = load_model('gender_model_15_55.h5', compile=False)
    age_model = load_model('age_model_15_55.h5', compile=False)
    race_model = load_model('race_model_15_55.h5', compile=False)

    # Accuracy
    x_train_age, x_test_age, y_train_age, y_test_age = train_test_split(images, ages, random_state=42)
    predictions = age_model.predict(x_test_age)
    y_pred = (np.rint(predictions)).astype(int)[:, 0]

    def accuracy(y_pred, y_test_age):
        acc = [1 for pred, age in zip(y_pred, y_test_age) if pred in range(age - 5, age + 5)]

        return sum(acc) / len(y_pred)

    print("Accuracy = ", accuracy(y_pred, y_test_age))

    x_train_gender, x_test_gender, y_train_gender, y_test_gender = train_test_split(images, genders, random_state=42)
    predictions = gender_model.predict(x_test_gender)
    y_pred = (predictions >= 0.5).astype(int)[:, 0]
    print("#####################")
    print("Accuracy = ", metrics.accuracy_score(y_test_gender, y_pred))

    # x_train_race, x_test_race, y_train_race, y_test_race = train_test_split(images, races, random_state=42)
    # score = race_model.evaluate(x_test_race, y_test_race, verbose=0)
    # print("#####################")
    # print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')

    # Prepare sample
    test_image = utils.load_image("testData/6.jpg")
    test_image = cv2.blur(test_image, (9, 9))
    test_image = utils.prepare_image_for_model_input(test_image)

    sample_to_predict = [test_image]
    sample_to_predict = np.array(sample_to_predict)

    # Prediction
    gender_prediction = gender_model.predict(sample_to_predict)
    age_prediction = age_model.predict(sample_to_predict)
    race_prediction = race_model.predict(sample_to_predict)

    lst = race_prediction[0].tolist()
    max_index = lst.index(max(lst))
    # print(Race(max_index).name)
    # for x in lst:
    #     print(x)

    # Print/show predictions
    gender = Gender.Male.name if gender_prediction[0] < 0.5 else Gender.Female.name
    age = int(np.rint(age_prediction[0]).astype(int))
    # race = Race(int(np.rint(race_prediction[0]).astype(int))).name
    race = Race(max_index).name

    print("Gender: {}, Age: {}, Race: {} => Predicted".format(gender, age, race))

    final_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
    plt.imshow(final_image)
    plt.axis('off')
    plt.title("Gender: {}, Age: {}, Race: {}".format(gender, age, race))
    plt.show()

    # Testing loop
    infinite_test(data, gender_model, age_model, race_model, 2)


def infinite_test(data, gender_model, age_model, race_model, sleep):
    time.sleep(sleep)

    for i in range(0, 10000):
        index = randrange(len(data.images))
        test_image = data.images[index]

        sample = [test_image]
        sample = np.array(sample)

        gender_prediction = gender_model.predict(sample)
        age_prediction = age_model.predict(sample)
        race_prediction = race_model.predict(sample)

        gender = Gender.Male.name if gender_prediction[0] < 0.5 else Gender.Female.name
        test1 = gender
        age = int(np.rint(age_prediction[0]).astype(int))
        lst = race_prediction[0].tolist()
        max_index = lst.index(max(lst))
        # race = Race(int(np.rint(race_prediction[0]).astype(int))).name
        race = Race(max_index).name

        print("Gender: {}, Age: {}, Race: {} => Predicted".format(gender, age, race))  # Predicted
        gender = "Male" if data.genders[index] < 0.5 else "Female"
        test2 = gender
        print(test1)
        print(test2)
        # if test1 == test2:
        #     continue
        predicted = "Gender: {}, Age: {}, Race: {} => Predicted".format(gender, age, race)
        real = "Gender: {}, Age: {}, Race: {} => Real".format(test1, data.ages[index], Race(data.races[index]).name)
        print("Gender: {}, Age: {}, Race: {} => Real".format(gender, data.ages[index], Race(data.races[index]).name))  # Real
        final_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
        plt.imshow(final_image)
        plt.axis('off')
        plt.title("{} \n{}".format(predicted, real))
        plt.show()
        time.sleep(sleep)
        # return


if __name__ == '__main__':
    main()
