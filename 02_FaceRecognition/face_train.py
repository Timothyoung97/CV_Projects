import os
import cv2 as cv
import numpy as np

# Create a list of people
people = []
for i in os.listdir(r'./Faces/train'):
    people.append(i)
    
DIR = r'./Faces/train'

# Using haar cascade
haar_cascade = cv.CascadeClassifier('haar_face.xml')

features = []
labels = []

# Create training data
def create_train():
    # Loop through each person in the training data
    for i, person in enumerate(people):
        path = os.path.join(DIR, person)
        label = i
        
        # Loop through each image in the person's folder
        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            img_array = cv.imread(img_path)
            if img_array is None:
                continue
            
            # Convert to grayscale
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)
            
            # Detect face
            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
            
            # Loop through each face
            for (x,y,w,h) in faces_rect:
                faces_roi = gray[y:y+h, x:x+w] # Region of interest
                features.append(faces_roi) # Add to features
                labels.append(label) # Add to labels

create_train()
print('Training done ------------')

# Convert features and labels to numpy arrays
feature = np.array(features, dtype='object')
labels = np.array(labels)

# Save features and labels
face_recognizer = cv.face.LBPHFaceRecognizer_create()

# Train the recognizer on the features list and the labels list
face_recognizer.train(feature, labels)

# Save the trained model
face_recognizer.save('face_trained.yml')
np.save('features.npy', features)
np.save('labels.npy', labels)