import cv2



face_cascade_name = 'haarcascade_frontalface_default.xml'

face_cascade = cv2.CascadeClassifier()

face_cascade.load(face_cascade_name)

face_id = input("Enter your id ")

cap = cv2.VideoCapture(0)

i = 0

for j in range(0,1000) :
    
    ret,frame = cap.read()

    frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(frame_gray)

    for (x,y,w,h) in faces :

        frame = cv2.rectangle(frame , (x,y) , (x+w,y+h), (0,255,255),3)

        ROI = frame_gray[y:y+h,x:x+w]
    
   
    cv2.imwrite(str(i)+'-'+str(face_id)+'.jpg',ROI)

    i = i+1

    cv2.imshow('detected face',frame)

   
    if cv2.waitKey(10) == 27:
        break
    

