from tensorflow.keras.models import load_model
import cv2,os
import numpy as np
import playsound as play

model = load_model('C:\\Users\\Sagar\\Final_Drowsiness_Detection\\model-010.model')

face_clsfr=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

source=cv2.VideoCapture(0)

labels_dict={0:'alert',1:'drowsy'}
color_dict={0:(0,255,0),1:(0,0,255)}
score=0
path="C:\\Users\\Sagar\\Desktop\\DAI\\Final_run"

while (True):

    ret, img = source.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_clsfr.detectMultiScale(gray, 1.3, 5)

    for x, y, w, h in faces:

        face_img = gray[y:y + w, x:x + w]
        resized = cv2.resize(face_img, (100, 100))
        normalized = resized / 255.0
        reshaped = np.reshape(normalized, (1, 100, 100, 1))
        result = model.predict(reshaped)

        label = np.argmax(result, axis=1)[0]

        if labels_dict[label] == 'drowsy':
            score = score + 1
        else:
            score = score - 1

        cv2.rectangle(img, (x, y), (x + w, y + h), color_dict[label], 2)
        cv2.rectangle(img, (x, y - 40), (x + w, y), color_dict[label], -1)
        cv2.putText(img, labels_dict[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        if score < 0:
            score = 0

        if score > 100:
            cv2.imwrite(os.path.join(path, 'image.jpg'), img)
            try:
                play.playsound('C:\\Users\\Sagar\\PycharmProjects\\DAI\\alarm.wav')
            except:
                pass


    cv2.imshow('LIVE', img)
    key = cv2.waitKey(1)

    if (key == 27):
        break

cv2.destroyAllWindows()
source.release()
