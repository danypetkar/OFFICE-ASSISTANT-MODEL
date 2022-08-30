# OFFICE ASSISTANT MODEL

## AIM AND OBJECTIVES:

### Aim

To create a model which can greet and converse lightly between a customer when they visit a shop or store and thus help reduce the load on the working staff and help customers with simple queries about the product and services offered by the store.

### Objectives:

The main objective of the project is to create a program which can be run on Jetson nano and start detecting using the camera module connected to the device.

Using appropriate data sets for recognizing and interpreting data using machine learning.

To show on the optical viewfinder of the camera module whether a given object is a Person and then greet and converse with them.

## ABSTRACT:
•	An object is classified based on whether it is a Person or not and then shown on the viewfinder of the camera.

•	I have completed this project on jetson nano which is a very small computational device.

•	A lot of research is being conducted in the field of Computer Vision and Machine Learning (ML), where machines are trained to identify various objects from one another. Machine Learning provides various techniques through which various objects can be detected.

•	A conversing AI is the need of the day as population increases and when the workforce is not able to manage peak hours in a store this AI becomes a helping hand for the workforce.

•	When a person is detected by the viewfinder of camera they are first greeted by the AI and then asked about how the AI can help in solving their problem.

## INTRODUCTION:

•	This project is based on Person detection and then converse with them model. We are going to implement this project with Machine Learning and this project will be run on jetson Nano.

•	This project can also be used to gather information about the needs of a customer and further increase our database about what a person will want in the future.

•	Person detection becomes difficult sometimes because of various sizes of People, also their distance from camera module as well as the time of the day which can make them harder for model to detect. And conversing with different people is hard as their needs are different.

•	Neural networks and machine learning have been used for these tasks and have obtained good results.

•	Machine learning algorithms have proven to be very useful in pattern recognition and classification, and hence can be used for Person recognition and conversations.

## LITERATURE REVIEW:

•	Conversational Commerce brings together online messaging like live chat and conversational AI together to deliver a refined shopping experience to customers 24/7. Customers are offered a personalized shopping experience like never before since the dawn of online shopping.  

•	For decades, traditional analytics have worked perfectly fine for the data-driven retail industry. However, Artificial Intelligence (AI) and Machine Learning (ML) have introduced an entirely new level of data processing which leads to deeper business insights. Data scientists could open a new world of possibilities to business owners extracting anomalies and correlations from hundreds of Artificial Intelligence/Machine Learning models.

•	Between 2013 and 2018, Artificial Intelligence startups raised $1.8 billion in 374 deals, according to CB Insights. Amazon can take credit for these impressive numbers because they made business leaders change their minds about Artificial Intelligence in the retail market – both physical stores and e-commerce strategies to stay ahead of the competition. At the moment over 28% of retailers are already deploying Artificial Intelligence/Machine Learning solutions, which is a sevenfold increase from 2016 when the number was only 4%.

•	In an era where consumers are constantly expecting personalized services, Conversational AI can be seen across the value chain. It is emerging as a powerful tool for retail businesses to help retailers align their offerings with the evolving customer expectations. Many brand leaders look at it as a solution to gain a strategic advantage over their competition. As a result, retail enterprises are increasingly leveraging the power of AI-driven data and emerging retail chatbot technologies to optimize customer experience, both online and offline. And the one consumer-oriented solution that retailers are relying on is Conversational AI.

•	Conversational AI can automate in-store operations and reduce a substantial amount of operational expenses in retail stores. It can help sales personnel to assist customers in the store, reduce queues through contactless payment methods, replenish stock by real-time stock monitoring, and overall improve the in-store experience for customers. A report by Aspect Software has revealed that 73% of customers prefer self-service options during their purchase journey and expect a faster checkout – a fact which does provide a strong impetus to retailers to implement Conversational AI solutions.

•	Digitally evolving multichannel retail has made store operations more complex. Store assistants are no longer restricted to receiving goods, stocking them, and managing inventories. They are expected to provide product information, keep track of promotional offers and handle merchandising for their customers. Conversational AI can be effectively utilized for such productive work, thereby saving a lot of time for your store agents to focus on other complex tasks. In simple terms, customers always expect a streamlined purchase journey, and Conversational AI chatbots steal the show in retail.

•	Conversational AI is also capable of detecting the mood, intent, and interest of your customers throughout the purchase journey. Some global retail brands have even introduced a facial recognition system for this function installed at checkout lanes. If a customer is annoyed, a store representative will immediately talk to him or her. Retail Chatbots also leverage the intent prediction feature to understand customers' tone, context, and behavior. It helps retailers build stronger relationships with customers by providing personalized assistance throughout the conversational flow.

•	Conversational AI can be deployed across different channels, hence providing brands an opportunity to create an omnichannel customer experience. These retail chatbots are capable of serving 24/7 and are significantly cheaper than onboarding more agents with rotational shifts. If the customer support query is complex or beyond the scope of the retail chatbot, there is a seamless process to hand off the query to a live agent based on their skill sets and current workload. This enables a smooth hassle-free customer experience for the support teams in the retail industry.

•	Over the last few years, news of the retail ‘e-pocalypse’ has spread far and wide. A string of public lay-offs and bankruptcies have plagued the sector in recent times. Store closure announcements that increased by over 200% in the last few years, were no different.

•	When it comes to retail, the online and offline worlds are both crucial and hence, need to be connected. Conversational AI plays a crucial role in bridging the gap between them – guaranteeing a uniform online and offline experience for customers. By engaging with retail chatbots, customers can browse through inventories and familiarize themselves with product offerings, and promotional deals, before they make it to the store. Thus, Conversational AI can contribute to a more positive and fulfilling in-store experience for customers, raising their overall engagement with retail brands and driving repeat visits.

•	The robotization of stores will result in reducing lines, lowering the number of human employees, and significant savings on operational expenses. 

## JETSON NANO COMPATIBILITY

•	The power of modern AI is now available for makers, learners, and embedded developers everywhere.

•	NVIDIA® Jetson Nano™ Developer Kit is a small, powerful computer that lets you run multiple neural networks in parallel for applications like image classification, object detection, segmentation, and speech processing. All in an easy-to-use platform that runs in as little as 5 watts.

•	Hence due to ease of process as well as reduced cost of implementation we have used Jetson nano for model detection and training.

•	NVIDIA JetPack SDK is the most comprehensive solution for building end-to-end accelerated AI applications. All Jetson modules and developer kits are supported by JetPack SDK.

•	In our model we have used JetPack version 4.6 which is the latest production release and supports all Jetson modules.

## OTHER DEVICES:


## PROPOSED SYSTEM:

1.	Study basics of machine learning and image recognition.

2.	Start with implementation

•	Front-end development

•	Back-end development

3.	Testing, analyzing and improvising the model. Machine learning libraries will be using machine learning to identify a object and then classify it according to whether it’s a Person or not.

4.	Use data sets to interpret the object and show on viewfinder with a bounding box annotation whether it is Person or not.

## METHODOLOGY:
The Office Assistant model is a program that focuses on implementing real time classification.

It is a prototype of a new product that comprises of the main module:

Jetson Nano

1. Object Detection

•	Ability to detect the location of Object in any input image or frame. The output is the bounding box coordinates on the detected Object.

•	This Data set identifies Object in a Bitmap graphic object and returns the bounding box image with annotation of name present.

2. Classification and process

•	Classification of the object based on whether it is Person or not on the viewfinder.

## INSTALLATION:


### Step 1: Data collection

For Data collection we are using OpenCV with Haarcascade to capture only Front Face.

```bash
    import cv2
    import glob
    import os
    cam = cv2.VideoCapture(1)
    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    count = 0
    while(True):
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(img, 1.3, 5)
        for (x,y,w,h) in faces:
            x1 = x+5
            y1 = y+5
            x2 = x+w
            y2 = y+h
            cv2.rectangle(img, (x1,y1), (x2,y2), (255,255,255), 2)     
        
             Save the captured image into the datasets folder
            cv2.imwrite("dataset/piyush" + str(count) + ".jpg", img[y1:y2,x1:x2])
            count += 1
        
            cv2.imshow('image', img)
        k = cv2.waitKey(200) & 0xff # Press 'ESC' for exiting video
        if k == 27:
            break
        elif count >= 350: # Take 30 face sample and stop video
            break
    cam.release()
    cv2.destroyAllWindows()
```


We need to change folder name every time for different-different person. And took 350 images for each Dany, Piyush and Nikhil.
And make labels.txt

##3 Step 2: Data Augmentation

For this project I applied gray-scale augmentation on some images using

### Step 3: Build the project directly on your Jetson.

```bash
    sudo apt-get remove --purge libreoffice*
    sudo apt-get remove --purge thunderbird*
    sudo fallocate -l 10.0G /swapfile1
    sudo chmod 600 /swapfile1
    sudo mkswap /swapfile1
    sudo vim /etc/fstab
    vim ~/.bashrc 
    export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1
    sudo apt-get update
    reboot
    sudo apt-get install git cmake
    git clone https://github.com/dusty-nv/jetson-inference
    cd jetson-inference/
    git submodule update --init
    sudo apt-get install libpython3-dev python3-numpy
    sudo apt-get install flac
    ls
    mkdir build
    cd build/
    sudo cmake ../
    sudo make
    sudo make install
    sudo ldconfig
    testing_____
    detectnet.py /dev/video0 
    cd jetson-inference/python/training/detection/ssd/
    mkdir models
    wget https://nvidia.box.com/shared/static/djf5w54rjvpqocsiztzaandq1m3avr7c.pth -O models/mobilenet-v1-ssd-mp-0_675.pth
```



### Step 4: After Testing, paste “my_data_450” folder which contain Images to train model into location “/jetson-inference/python/training/classification/data”

### Step 5: Training Image classification Model using ResNet18.

```bash
    cd  /jetson-inference/python/training/classification/
    python3 train.py --model-dir=models/my_data_450 --batch-size=4 --workers=1 --epochs=50 data/my_data_450
    Now convert the model to ONNX
    python3 onnx_export.py –model-dir=models/my_data_450
    Now testing model using webcam
    imagenet.py –model=models/my_data_450/resnet18.onnx –input_blob=input_0 –output_blob=output_0 –labels=data/my_data_450/labels.txt /dev/video0
```
### Step 6: Detection Using OpenCV

```bash
    sudo pip3 install speechrecognition
    sudo pip3 install sounddevice pyaudio google-api-python-client
    Now copy haarcascade_frontalface_default.xml file into “/jetson-inference/python/training/classification/”
    Make python file aaa.py
```
### jetson for
```bash
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
                            ans = 'how can i help you'
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


```

### Step 7: Run final Project
```bash
    python3 aaa.py
```
## ADVANTAGES:
•	AI chatbots provide an even higher level of customer service, improve searching, send notifications about new collections, and suggest similar products.

•	Retailers also invest in technologies that help customers in the shopping process and also help staff in stores. At the same time, it helps with inventory management thanks to real-time monitoring capabilities.

•	Applications of AI for retail stores could help businesses set prices for their products, visualizing the likely outcomes of multiple pricing strategies.

•	Leftovers and out-of-stock scenarios can be eliminated. AI in the retail supply chain can be used for restocking — calculating the demand for a particular product by taking into account a history of sales, location, weather, trends, promotions, and other parameters.

•	It can work around the clock and therefore can detect all day long.

•	When completely automated no user input is required and therefore works with absolute efficiency and speed.

## APPLICATION:

•	Detects an object and then shows whether the object is Person or not in each image frame or viewfinder using a camera module and then begins conversing.

•	According to a retail executives survey by Capgemini at the AI in Retail Conference, the application of the technology in retail could save up to $340 billion each year for the industry by 2020. An estimated 80% of these savings will come from AI’s improvement of supply chain management and returns.

•	Can be used as a reference for other ai models based on Office Assistant Model

## FUTURE SCOPE:
•	As we know technology is marching towards automation, so this project is one of the step towards automation.

•	Thus, for more accurate results it needs to be trained for more images, and for a greater number of epochs.

•	Office Assistant model can be very easily implemented in places where workforce is sparse.

•	Conversational commerce could become all the more prevalent in the metaverse, the virtual reality representation of the internet, with voice-enabled shopping potentially accounting for 30% of all ecommerce revenues by 2030.

•	Office Assistant model can be further improved by adding more images and conversational data of different People with newer models to further improve the detection and hence be future ready.


## CONCLUSION:

•	In this project our model is trying to detect a Person and then begin conversing with them.

•	This model tries to solve the problem of places where Workforce gets easily overwhelmed by the number of customer’s every day at peak hours and hence can be of great help there.

•	The model is efficient and highly accurate and hence works without any lag and also as the data is already exported to model folder can be made to work offline.

