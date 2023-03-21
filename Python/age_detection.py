#Reference https://github.com/spmallick/learnopencv

import cv2 as cv
import time
from fer import FER
from os import read
import pyrebase
from pyrebase.pyrebase import initialize_app

config = {
  "apiKey": "AIzaSyCyV2Lgv2slQaKlIp3kDPdVCYZmb26DT08",
  "authDomain": "techin510-lab9-fefb4.firebaseapp.com",
  "projectId": "techin510-lab9-fefb4",
  "databaseURL": "https://techin510-lab9-fefb4-default-rtdb.firebaseio.com/",
  "storageBucket": "techin510-lab9-fefb4.appspot.com",
  "messagingSenderId": "707443400713",
  "appId": "1:707443400713:web:3edb753bbafc2e846918d8"
}

firebase = pyrebase.initialize_app(config)

milestoneDB = firebase.database()

def initializeApp():
    initializeDictionary = { "is_child_exist": False, "user_in_front" : False, "is_engaging": True }
    milestoneDB.child("milestone2").set(initializeDictionary)


initializeApp()

# For Emotion Detection
detector = FER()
boring_threshold = 80
counter = 0
emotion_counter=[]

def getFaceBox(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
            cv.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn, bboxes


faceProto = "model/opencv_face_detector.pbtxt"
faceModel = "model/opencv_face_detector_uint8.pb"

ageProto = "model/age_deploy.prototxt"
ageModel = "model/age_net.caffemodel"


MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

# Load network
ageNet = cv.dnn.readNet(ageModel, ageProto)
faceNet = cv.dnn.readNet(faceModel, faceProto)

# Use GPU
ageNet.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
ageNet.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)

# Get Video File
cap = cv.VideoCapture(0)
padding = 20

faceDetectionCounter = 0
faceDetectionThreshold = 10

while cv.waitKey(1) < 0:
    t = time.time()

    hasFrame, frame = cap.read()
    frame = cv.flip(frame,1)
    
    if not hasFrame:
        cv.waitKey()
        break

    frameFace, bboxes = getFaceBox(faceNet, frame)


    if len(bboxes) == 0:
        milestoneDB.child("milestone2").update({"user_in_front" : False})
    else:
        milestoneDB.child("milestone2").update({"user_in_front" : True})
    
    # if faceDetectionCounter >= faceDetectionThreshold:
        # milestoneDB.child("milestone2").update({"user_in_front" : False})

    # if len(faceDetection) < faceDetectionThreshold:
    #     faceDetection.append(len(bboxes))
    # else:
    #     emotion_counter[faceDetectionCounter] = len(bboxes)
    
    


    # if faceDetection.count(0) >= faceDetectionThreshold:
    #     milestoneDB.child("milestone2").update({"user_in_front" : False})
    # else:
    #     milestoneDB.child("milestone2").update({"user_in_front" : True})

    # if len(faceDetection) >= faceDetectionThreshold:
    #     faceDetection = []
    
    print(len(bboxes))
    
    # Emotion Detection Starts

    result = detector.detect_emotions(frameFace)

    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]        # Detecting Emotion

    for a in result:
        box = a["box"]
        if len(box) > 0:
            x1 = box[0]
            x2 = box[0]+box[2]

            y1 = box[1]
            y2 = box[1]+box[3]
            # cv.rectangle(frameFace, (x1,y1),(x2,y2),(0,255,0),8)
            label = max(a["emotions"], key= a["emotions"].get)
            cv.putText(frameFace, label, (x1, y1-10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv.LINE_AA)
            # cv.putText(frameFace, label, (bbox[0], bbox[1]), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv.LINE_AA)

            if len(emotion_counter) < boring_threshold:
                emotion_counter.append(label)
            else:
                emotion_counter[counter] = label

    # # neutral counter
    neutral_counter = emotion_counter.count('neutral')

    message = f"number of frame when neutral emotion = {neutral_counter}/{boring_threshold}"

    video_size = frame.shape

    if len(emotion_counter) < boring_threshold:
        cv.rectangle(frameFace, (0,video_size[0]-100), (frameFace.shape[1],frameFace.shape[0]),(0,255,0),thickness=-1)
        message = f"Please wait for data gathering --- {len(emotion_counter)}/{boring_threshold}"     
    elif neutral_counter > boring_threshold * 0.7:
        cv.rectangle(frameFace, (0,video_size[0]-100), (frameFace.shape[1],frameFace.shape[0]),(0,0,255),thickness=-1)
        message = "Alert : Child might be boring --- " + message
        
        #send data
        datas = {"is_engaging": False}
        milestoneDB.child("milestone2").update(datas)
        #reset
        emotion_counter = []

    else:
        cv.rectangle(frameFace, (0,video_size[0]-100), (frameFace.shape[1],frameFace.shape[0]),(255,0,0),thickness=-1)
        message = "Child is engaging! --- " + message

    # draw debug window
    # cv.rectangle(frame, (0,video_size[0]-100), (frame.shape[1],frame.shape[0]),(255,0,0),thickness=-1)
    cv.putText(frameFace,message,(50,video_size[0]-50), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv.LINE_AA)

    counter = counter + 1
    if counter == boring_threshold:
        counter = 0


    # # Emotion Detection Ends









    child_or_parents = {"child":0, "adult":0}
    for bbox in bboxes:
        face = frame[max(0,bbox[1]-padding):min(bbox[3]+padding,frame.shape[0]-1),max(0,bbox[0]-padding):min(bbox[2]+padding, frame.shape[1]-1)]
        blob = cv.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

        # Detecting Age
        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]
        # print("Age Output : {}".format(agePreds))
        # print("Age : {}, conf = {:.3f}".format(age, agePreds[0].max()))
        # print(bboxes)

        if agePreds[0].argmax() <= 3:
            child_or_parents["child"] += 1
        else :
            child_or_parents["adult"] += 1

        label = "Age : " + "{}".format(age) 
        cv.putText(frameFace, label, (bbox[0], bbox[3] + 25), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv.LINE_AA)
        

    # draw children debug window
    video_size = frame.shape

    num_child = child_or_parents["child"]
    num_adult = child_or_parents["adult"]

    if child_or_parents["child"] > 0 :    
        message = f"child detected : # of children : {num_child}, # of adults : {num_adult}"
        cv.rectangle(frameFace, (0,video_size[0]-200), (frame.shape[1],frame.shape[0]-100),(255,0,0),thickness=-1)

        milestoneDB.child("milestone2").update({"is_child_exist" : True})
    else :
        milestoneDB.child("milestone2").update({"is_child_exist" : False})
        message = f"No child detected : # of children : {num_child}, # of adults : {num_adult}"
        cv.rectangle(frameFace, (0,video_size[0]-200), (frame.shape[1],frame.shape[0]-100),(0,0,255),thickness=-1)

    cv.putText(frameFace,message,(50,video_size[0]-150), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv.LINE_AA)




    # cv.imshow("face_detection",frameFace)
    # print("time : {:.3f}".format(time.time() - t))



