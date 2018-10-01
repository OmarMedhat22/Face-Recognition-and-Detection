import cv2

import numpy as np

recognizer = cv2.face.LBPHFaceRecognizer_create()

recognizer.read('trained_model.yml')

cascadePath = "haarcascade_frontalface_default.xml"

faceCascade = cv2.CascadeClassifier(cascadePath)

font = cv2.FONT_HERSHEY_SIMPLEX

cam = cv2.VideoCapture(0)


while True:

    ret,im = cam.read(0)

    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray,1.2,5)

    for(x,y,w,h) in faces:

        cv2.rectangle(im,(x,y),(x+w,y+h),(255,0,255),2)

        face_id,conf = recognizer.predict(gray[y:y+h,x:x+w])

        if (face_id == 1):

            face_id = ";dl;dsg;sfg;sfg"
            cv2.putText(im , str(face_id), (x,y-40),font,2,(255,255,255),3)

        else:

            face_id = "unknown"
            cv2.putText(im , str(face_id), (x,y-40),font,2,(255,255,255),3)

    cv2.imshow('image',im)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroySllWindows()


            
            

        
