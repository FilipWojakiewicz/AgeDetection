import cv2
import os
from dataclasses import dataclass
from typing import Any


class FaceData:
    def __init__(self):
        # self.images: list[SingleFaceData] = []
        self.images = []
        self.ages = []
        self.genders = []
        self.races = []

    def load_images(self):
        path = "UTKFace/"
        character = '_'
        ageMIN = 20
        ageMAX = 76
        for filename in os.listdir(path):
            age = filename.split(character)
            if int(age[0]) >= ageMIN and int(age[0]) <= ageMAX:
                img = cv2.imread(os.path.join(path, filename), cv2.IMREAD_COLOR)
                # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # print(filename)
                if img is not None:
                    split = filename.split("_")
                    self.images.append(img)
                    self.ages.append(int(split[0]))
                    self.genders.append(int(split[1]))
                    self.races.append(int(split[2]))


@dataclass
class SingleFaceData:
    image: Any
    age: int
    gender: int
    race: int
