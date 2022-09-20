import os
import numpy as np
import cv2 as cv

haar_cascade = cv.CascadeClassifier('haar_face.xml')

people = []
for i in os.listdir(r'./Faces/train'):
    people.append(i)

# features = np.load('./features.npy', allow_pickle=True)
# labels = np.load('./labels.npy', allow_pickle=True)

face_recognizer = cv.face.LBPHFaceRecognizer_create() # Local Binary Patterns Histograms
face_recognizer.read('face_trained.yml') # Load the trained model

img = cv.imread(r'./Faces/val/ben_afflek/1.jpg') # Load the image

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) # Convert to grayscale

cv.imshow('Person', gray)

faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4) # Detect face

for (x,y,w,h) in faces_rect:
    faces_roi = gray[y:y+h, x:x+w] # Region of interest
    
    label, confidence = face_recognizer.predict(faces_roi) # Predict the label of the image
    print(f'Label = {people[label]} with a confidence of {confidence}') # Print the label
    
    cv.putText(img, str(people[label]), (20,20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), thickness=2) # Draw the label on the image
    cv.rectangle(img, (x,y), (x+w, y+h), (0,255,0), thickness=2) # Draw the rectangle around the face
    
cv.imshow('Detected Face', img)

cv.waitKey(0)