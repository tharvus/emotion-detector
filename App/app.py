import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

PATH = "../Model/my_model.h5"

# store all labels
all_labels = {
    0: "happy",
    1: "neutral",
    2: "sad",
}


# model = tf.keras.Sequential([])
# model.load_weights("checkpoint")
model = load_model(PATH)

video = cv2.VideoCapture(0)

# create face classifier
face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def detect_face_and_predict_emotion(vid):
    # convert image to greyscale
    gray_img = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)

    # detect multiple images using haarcascade classifier
    faces = face_classifier.detectMultiScale(
        gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
    )

    for x, y, w, h in faces:
        # determine the face
        face_region_of_interest = gray_img[y : y + h, x : x + h]

        # resize the shape of the face so it can be fitted to the model
        resized_face = cv2.resize(
            face_region_of_interest, (48, 48), interpolation=cv2.INTER_AREA
        )

        # normalise the face according to model requirements
        normalised_face = resized_face / 255.0
        # Since its a greyscale image, duplicate the channel across 3 channels
        normalised_face = np.dstack([normalised_face] * 3)
        # reshape the face so it can be inputted to the model
        reshaped_face = normalised_face.reshape(1, 48, 48, 3)

        # predict image emotion
        pred_tensor = model.predict(reshaped_face)
        # determine index with highest value
        pred_index = np.argmax(pred_tensor)
        # determine probability
        pred_value = np.max(pred_tensor)
        # determine the emotion by retriving the label
        emotion_predicted = all_labels[pred_index]

        # draw a green box over the face with a thickness of 4
        cv2.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 4)

        # put the emotion predicted
        cv2.putText(
            vid,
            emotion_predicted + " " + str(pred_value),
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
        )

    return faces


# keep reading input until user presses the q key
while True:
    result, video_frame = video.read()

    faces = detect_face_and_predict_emotion(video_frame)
    cv2.imshow("Face Detection using Haars Cascades", video_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# stop application by closing all windows
video.release()
cv2.destroyAllWindows()
