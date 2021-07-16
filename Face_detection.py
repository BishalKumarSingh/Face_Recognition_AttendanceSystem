# Importing all the libraries

import face_recognition # This find the faces in our images . This is done using HOG (Histogram of Oriented Gradients) at the backend.
import cv2
import numpy as np
import os
import datetime

path = 'C:\\Users\\bisha\\Desktop\\Face Detection\\Photo'
images = []     # LIST CONTAINING ALL THE IMAGES
className = []    # LIST CONTAINING ALL THE CORRESPONDING CLASS Names
myList = os.listdir(path)
print("Total Classes Detected:",len(myList))
for x,cl in enumerate(myList):
        curImg = cv2.imread(f'{path}/{cl}')
        images.append(curImg)
        className.append(os.path.splitext(cl)[0])
        
def markAttendance(name):
    with open('C:\\Users\\bisha\\Desktop\\Face Detection\\Attendance.csv','r+') as f:
        myDataList = f.readlines()
        nameList =[]
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in  nameList:
            now = datetime.datetime.now()
            t_string = now.strftime("%H:%M:%S")
            d_string = now.strftime("%d-%m-%y")
            f.writelines(f'\n{name},{d_string},{t_string}')

# Computing Encoding

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

# Now we can simply call this function with the images list as the input arguments.

encodeListKnown = findEncodings(images)
print('Encodings Complete')

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    cv2.imshow('Webcam1',imgS)
    if cv2.waitKey(20) & 0xFF==ord('d'):
        break
    #cap.release()
    #cv2.destroyAllWindows
    
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

        matchIndex = np.argmin(faceDis)

        '''if matches[matchIndex]:
            name = className[matchIndex].upper()
            print(name)
            y1,x2,y2,x1 = faceLoc
            y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            cv2.imshow('webcam2',img)
            markAttendance(name)'''
        if faceDis[matchIndex] < 0.50:
            name = className[matchIndex].upper()
            markAttendance(name)
        else:
            name = 'Unknown'
        #print(name)
        y1,x2,y2,x1 = faceLoc
        y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
        cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
        cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
        cv2.imshow('Webcam',img)





'''if matches[matchIndex]:
    name = classNames[matchIndex].upper()
    #print(name)
    y1,x2,y2,x1 = faceLoc
    y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
    cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
    cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
    cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
    markAttendance(name)'''


'''if faceDis[matchIndex]< 0.50:
    name = classNames[matchIndex].upper()
    markAttendance(name)
else: name = 'Unknown'
#print(name)
y1,x2,y2,x1 = faceLoc
y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)'''

cv2.waitKey(0)