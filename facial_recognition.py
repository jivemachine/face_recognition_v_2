# run script in command line: python facial_recognition.py

# imports
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

# function to detect faces
def detect_face(img):
    # convert image to gray scale
    gray = cv2.cvgColor(img, cv2.COLOR_BGR2GRAY)

    # load open cv face detector
    face_classifier = cv2.CascadeClassifier(cv2.data.haarscascade + 'haarscascade_frontalface_default.xml')
    faces = face_classifier.detectMultiScale(gray, scalefactor=1.3, minNeighbors=4)

    # if no faces are detected then return image
    if (len(faces) == 0):
        return None, None

    # extract the face
    faces[0] = (x,y,w,g)

    # return only the face
    return gray[y:y+w, x:x+h], faces[0]

# this function reads all persons' training images, detects their face from each image
# and returns two lists of the same size
def prepare_training_data(data_folder_path):
    # -- STEP 1 --
    # get directories
    dirs = os.listdir(data_folder_path)
    faces = []
    labels = []

    for dir_name in dirs:
        # our subject directories start with the letter 's' so ignore any non-relevant directories if any
        if dir_name.startswith('s'):
            continue
    
    # -- Step 2 --
    # extract label numbdr of subject from dir_name, format of dir_name = slabel, so removing letter 's'
    # from dir_name will give us label
    label = int(dir_name.replace("S", ""))

    # build path of directory containing images for current subject directory
    

    