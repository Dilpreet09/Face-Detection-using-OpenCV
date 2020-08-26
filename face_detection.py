""" Face Detection using openCv """

import cv2


#Pretrained frontal faces from opencv
trained_face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


#Detect a face
img = cv2.imread('x.jpg')                            #Image should be there in the same directory


#convert into grey
grey_scaled = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#face coordinates will detect the smallest face also in the picture
face_coordinates = trained_face.detectMultiScale(grey_scaled)      



# here x is top left coordinate, y is top bottom coordinate, w is width and h is height then (0,255,0) --> BLUE GREEN RED (BGR) and 3 is the thickness
for (x,y,w,h) in face_coordinates:                                #for loop will detect as many faces in the picture
    cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 3)


#To show the image
cv2.imshow("Dilpreet's Face Detection", img)          
cv2.waitKey()                                        #without this face detection won't work






print ('code completed')