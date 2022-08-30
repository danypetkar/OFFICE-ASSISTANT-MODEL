class_name = ['Chetan','Sanket']
#testing cascad
import cv2
import numpy as np
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
font = cv2.FONT_HERSHEY_SIMPLEX
# To capture video from webcam. 
cap = cv2.VideoCapture(0)
# To use a video file as input 
# cap = cv2.VideoCapture('filename.mp4')
while True:
    # Read the frame
    _, img = cap.read()

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    test = img
    test = cv2.resize(test,(150,150)) #(150,150,3)
    test = test.reshape(1,150,150,3)
    iden = []
    pred = model.predict(test)
    iden.append(int(pred[0][0]))
    print(iden)
    

    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    #print(faces.shape)
    # Draw the rectangle around each face
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(img, str(class_name[int(pred[0][0])]), (x+5,y-5), font, 1, (255,0,0), 4)

    # Display
    cv2.imshow('face Recognition', img)

    # Stop if escape key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
# Release the VideoCapture object
cap.release()
cv2.destroyAllWindows()
