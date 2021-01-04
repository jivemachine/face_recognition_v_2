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