import cv2


class Utils:
    @staticmethod
    def prepare_image_for_model_input(image):
        cropped_image = image
        face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        for (x, y, w, h) in faces:
            cropped_image = image[y:y+w, x:x+h]
            # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cropped_image = cv2.resize(cropped_image, (200, 200))
        # cv2.imshow("Faces found", cropped_image)
        # cv2.waitKey(0)
        return cropped_image

    @staticmethod
    def load_image(path):
        return cv2.imread(path, cv2.IMREAD_COLOR)

