import cv2


face_cascade_name = 'haarcascade_frontalface_default.xml'


face_cascade = cv2.CascadeClassifier()

face_cascade.load(face_cascade_name)


cap = cv2.VideoCapture(0)

while True:

    ret,frame = cap.read()
    
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(frame)


    for (x,y,w,h) in faces:

        
        frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),3)
        #frame = cv2.circle(frame,(x+w//2,y + int(h/2)),int(h/2),(255,0,255),4)
        frame = cv2.ellipse(frame,(x+w//2,y + int(h/2)),(w//2,h//2),0,0,360,(0,255,0),4)

    cv2.imshow("Capture",frame)

    if cv2.waitKey(10) == 27:
        break
    
        
        
