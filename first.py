# -*- coding: utf-8 -*-
"""
Created on Sun Aug 2  10:37:25 2020

@author: Vishv
"""

"""
Photo Face Detection
"""
import cv2

face_cascade = cv2.CascadeClassifier('.\\pretrained_models\\haarcascade_frontalface_default.xml')

img = cv2.imread('demo_img.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.3, 5)
#print(faces)
for (x,y,w,h) in faces:
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)

cv2.imwrite('image.png',img)
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()


