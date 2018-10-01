import cv2
import numpy as np
import os


face_images = []

face_id = []

recoginzer = cv2.face.LBPHFaceRecognizer_create()

def train_images():

    path = r"C:\Users\omar\pvideos\face\face_detection\dataset"


    for image_path in os.listdir(path):

        input_path = os.path.join(path,image_path)

        img = cv2.imread(input_path,0)

        image = np.array(img,'uint8')

        #print(image)

        print(input_path)

        i = input_path[-5]

        print(i)

        face_images.append(image)
        face_id.append(int(i))

        cv2.imshow('frame',img)

        if cv2.waitKey(10) & 0xFF==ord('q'):
            break
            cv2.destroyAllWindows()


    return face_images,face_id
    

        


face_images,face_id =  train_images()

recoginzer.train(face_images,np.array(face_id))
recoginzer.write('trained_model.yml')

cv2.waitKey(1000)
cv2.destroyAllWindows()
