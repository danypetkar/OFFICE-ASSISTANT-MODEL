# jetson for
import speech_recognition as sr
from gtts import gTTS
import os
import playsound
import jetson.inference
import jetson.utils
import cv2
import glob
import os
import time
net = jetson.inference.imageNet('',['--model=models/my_data_450/resnet18.onnx','--labels=data/my_data_450/labels.txt','--input_blob=input_0','--output_blob=output_0'])
cam = cv2.VideoCapture(0)

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
font = cv2.FONT_HERSHEY_SIMPLEX
count = 0
identify = []
while(True):
    ret, img = cam.read()
    img = cv2.resize(img,(500,500))
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(img, 1.3, 5)
    
    for (x,y,w,h) in faces:
        x1 = x
        y1 = y
        x2 = x+w
        y2 = y+h
        test = img[y1:y2,x1:x2]
        test = jetson.utils.cudaFromNumpy(test)
        class_idx, confidence = net.Classify(test)
        class_desc = net.GetClassDesc(class_idx)
        #print('JIO MERE',class_desc)
        cv2.rectangle(img, (x1,y1), (x2,y2), (255,255,255), 2)
        cv2.putText(img, str(str(round(confidence*100)) + class_desc), (x+5,y-5), font, 1, (255,0,0), 4)
        count += 1
        
        cv2.imshow('image', img)
        ##############
        if confidence > 0.99:
            identify.append(class_desc)
            if len(identify)>20:
                value = identify[10]
                text = "Good Morning"+' '+str(value)
                tts=gTTS(text=text,lang="en")
                filename = "test.mp3"
                tts.save(filename)
                playsound.playsound(filename)
                os.remove(filename)
                ##############################################
                os.system("arecord -D hw:2,0 -c 1 -r 16000 -f S16_LE -d 5 cap3.wav")
                r = sr.Recognizer()
                with sr.AudioFile('cap3.wav') as source:
                    audio_text = r.listen(source)
                    try:
                        text = r.recognize_google(audio_text)
                        print(text)
                        if text == 'good morning':
                            ans = 'have a nice day'
                        elif text == 'how are you':
                            ans = 'i am fine'
                        else:
                            ans = 'teri saale nikal'
                        tts=gTTS(text=ans,lang="en")
                        filename = "ans.mp3"
                        tts.save(filename)
                        playsound.playsound(filename)
                        os.remove(filename)
                    except:
                        print('Sorry.. run again...')

                #print(text)
                identify = []
        #print(identify)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    #elif count >= 100: # Take 100 face sample and stop video
         #break
cam.release()
cv2.destroyAllWindows()
