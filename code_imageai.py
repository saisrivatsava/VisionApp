from imageai.Detection import ObjectDetection
from imageai.Prediction import ImagePrediction

import os
import argparse
import sys
# import queue 
from multiprocessing import Process
import multiprocessing
import time
import ray

import cv2

from datetime import datetime
startTime = datetime.now()
# https://github.com/OlafenwaMoses/ImageAI/tree/master/imageai/Detection

# ray.init()

detection_flag = 0


detected_frames_list = []



def drawPerson(frame, box):
    # Draw a bounding box.
    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255, 178, 50), 2)
    return frame

def getDetectorObject(execution_path, model):
    
    detector = ObjectDetection()


    if model == "yolo":
        print("Using YOLO3 for Detection")
        detector.setModelTypeAsYOLOv3()
        detector.setModelPath(os.path.join(execution_path,"models/yolo.h5"))
    elif model == "yolo-tiny":
        print("Using YOLO-TINY for Detection")
        detector.setModelTypeAsTinyYOLOv3()
        detector.setModelPath(os.path.join(execution_path,"models/yolo-tiny.h5"))
    elif model == "ratina":
        print("Using RatinaNet for Detection")
        detector.setModelTypeAsRetinaNet()
        detector.setModelPath(os.path.join(execution_path,"models/resnet50_coco_best_v2.0.1.h5"))

    detector.loadModel(detection_speed="fast")
    return detector



# @ray.remote
def loadVideoAndDetectIntruder(execution_path, detector, video):
    lastUploaded = datetime.now()

    cap = cv2.VideoCapture(os.path.join(execution_path, "data/{}.mp4".format(video)))

    
    print("width : {}".format(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    print("height : {}".format(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    frame_num = 0
    intruder_frames_count = 0


    while cap.isOpened():
        ret, frame1 = cap.read()
        ret, frame2 = cap.read()

        if not ret:
            break

        diff = cv2.absdiff(frame1, frame2)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)

        _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
        dilated  = cv2.dilate(thresh, None, iterations=3)
        contours,_ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)
        
            if cv2.contourArea(contour ) < 400:
                continue

            # print("===Motion Detected===")
            frame_detected = frame1
            custom_objects = detector.CustomObjects(person=True, motorcycle=True)

        # detections = detector.detectCustomObjectsFromImage(custom_objects=custom_objects, input_image=frame_detected, input_type="array",output_image_path=os.path.join(execution_path , "detected/detected_img_{}.jpg".format(frame_num)),minimum_percentage_probability=60)

            detected_image_array, detections = detector.detectCustomObjectsFromImage(output_type="array", custom_objects=custom_objects, input_image=frame_detected, input_type="array")#,minimum_percentage_probability=30)

            for eachObject in detections:
                print(eachObject["name"] , " : ", eachObject["percentage_probability"], " : ", eachObject["box_points"] )
                print("--------------------------------")
                intruder_frames_count +=1
                if eachObject["name"] == "person" and eachObject["percentage_probability"] > 40 :
                    if (datetime.now() - lastUploaded).seconds >= 15 or intruder_frames_count > 5: 
                        intruder_frames_count = 0

                        processDetections(detected_image_array)
                        lastUploaded = datetime.now()
                        cv2.imshow("person", detected_image_array)#drawPerson(detected_image_array, eachObject["box_points"]))

            
        frame_num= frame_num+1
    # print("frame no :"+str(frame_num))

        cv2.imshow("survilence", frame1)

        frame1 = frame2
        ret, frame2 = cap.read()

        if cv2.waitKey(40)==27:
            break

    detection_flag = -1

    cv2.destroyAllWindows()
    cap.release()

    print("Number of Frames : "+ str(frame_num))
    print("Persons Detected : "+ str(persons))
    print("Total Time taken : "+str(datetime.now() - startTime))
    # print(datetime.now() - startTime)




# @ray.remote
def processDetections(detected_image_array):
    print("=========Image came to upload===============")
    ounter = 0

    # while True:
    #     # frames_queue.empty():
    #     if detection_flag == -1:
    #         break
    #     if detection_flag == 0:
    #         print("Waiting for Frames to process")
    #         time.sleep(10)
    #     # break
    #     if not len(detected_frames_list) == 0:
    #         print("person detected : "+ str(ounter))
    #         ounter+=1
    #         frame = detected_frames_list[0]
    #         detected_frames_list.pop(0)
    #     else:
    #         print("Empty list")
            
    # cap.release()


def showFrame(frame_array):
    cv2.imshow("person", frame_array)
    cv2.waitKey(10)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    ray.init()

    manager = multiprocessing.Manager()

    # Define a list (queue) for tasks and computation results
    # tasks = manager.Queue()
    # results = manager.Queue()

    frames_queue = manager.Queue(maxsize=1000) 
    pool = multiprocessing.Pool(processes=2)

    execution_path = os.getcwd()
    parser = argparse.ArgumentParser(description='Intruder Detection ...')
    parser.add_argument('--video', help='Video Name.')
    parser.add_argument('--model', help='Model Name.')
    args = vars(parser.parse_args())

    model = args["model"]
    video = args["video"]

    detector = getDetectorObject(execution_path, model)
    loadVideoAndDetectIntruder(execution_path, detector, video)
    # processDetections(frames_queue)
    # detector_process = multiprocessing.Process(target=loadVideoAndDetectIntruder, args=(execution_path, detector, video, frames_queue))
    # detector_process.start()
    # ray.get([loadVideoAndDetectIntruder.remote((execution_path), (detector), (video)), processDetections.remote()]) 

    



    # ray.get([loadVideoAndDetectIntruder.remote(execution_path, detector, video, frames_queue), processDetections.remote(frames_queue)])

    # p1 = Process(target = loadVideoAndDetectIntruder, args=((execution_path), (detector), (video), ))
    # p1.start()
    # p2 = Process(target = processDetections)
    # p2.start()
    # p1.join()
    # p2.join()
    # target=reader_proc, args=((pqueue),)