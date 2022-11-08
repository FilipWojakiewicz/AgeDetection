import cv2
import numpy as np
import matplotlib.pyplot as plt
from data import FaceData
from train import Train, Gender, Race
from utils import Utils
from keras.models import load_model
from random import randrange
import time
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
    # train.train_age_model(images, ages)
    # train.train_gender_model(images, genders)
    # train.train_race_model(images, races)

    # Load model
    gender_model = load_model('gender_model.h5', compile=False)
    age_model = load_model('age_model.h5', compile=False)
    race_model = load_model('race_model.h5', compile=False)

    # Prepare sample
    test_image = utils.load_image("testData/testFace3.jpg")
    test_image = utils.prepare_image_for_model_input(test_image)

    sample_to_predict = [test_image]
    sample_to_predict = np.array(sample_to_predict)

    # Prediction
    gender_prediction = gender_model.predict(sample_to_predict)
    age_prediction = age_model.predict(sample_to_predict)
    race_prediction = race_model.predict(sample_to_predict)

    # Print/show predictions
    gender = Gender.Male.name if gender_prediction[0] < 0.5 else Gender.Female.name
    age = int(np.rint(age_prediction[0]).astype(int))
    race = Race(int(np.rint(race_prediction[0]).astype(int))).name

    print("Gender: {}, Age: {}, Race: {} => Predicted".format(gender, age, race))

    final_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
    plt.imshow(final_image)
    plt.axis('off')
    plt.title("Gender: {}, Age: {}, Race: {}".format(gender, age, race))
    plt.show()

    # Testing
    infinite_test(data, gender_model, age_model, race_model, 5)


def infinite_test(data, gender_model, age_model, race_model, sleep):
    time.sleep(sleep)

    for i in range(0, 100):
        index = randrange(len(data.images))
        test_image = data.images[index]

        sample = [test_image]
        sample = np.array(sample)

        gender_prediction = gender_model.predict(sample)
        age_prediction = age_model.predict(sample)
        race_prediction = race_model.predict(sample)

        gender = Gender.Male.name if gender_prediction[0] < 0.5 else Gender.Female.name
        age = int(np.rint(age_prediction[0]).astype(int))
        race = Race(int(np.rint(race_prediction[0]).astype(int))).name

        print("Gender: {}, Age: {}, Race: {} => Predicted".format(gender, age, race))  # Predicted
        gender = "Male" if data.genders[index] < 0.5 else "Female"
        print("Gender: {}, Age: {}, Race: {} => Real".format(gender, data.ages[index], Race(data.races[index]).name))  # Real
        final_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
        plt.imshow(final_image)
        plt.axis('off')
        plt.title("Gender: {}, Age: {}, Race: {}".format(gender, age, race))
        plt.show()
        time.sleep(sleep)


if __name__ == '__main__':
    main()
