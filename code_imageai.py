from imageai.Detection import ObjectDetection
from imageai.Prediction import ImagePrediction

import os
import argparse
import sys

import cv2

# https://github.com/OlafenwaMoses/ImageAI/tree/master/imageai/Detection

execution_path = os.getcwd()
detector = ObjectDetection()


parser = argparse.ArgumentParser(description='Intruder Detection ...')
parser.add_argument('--video', help='Video Name.')
parser.add_argument('--model', help='Model Name.')
args = vars(parser.parse_args())


if args["model"] == "yolo":
    print("Using YOLO3 for Detection")
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath(os.path.join(execution_path,"models/yolo.h5"))
elif args["model"] == "yolo-tiny":
    print("Using YOLO-TINY for Detection")
    detector.setModelTypeAsTinyYOLOv3()
    detector.setModelPath(os.path.join(execution_path,"models/yolo-tiny.h5"))
elif args["model"] == "ratina":
    print("Using RatinaNet for Detection")
    detector.setModelTypeAsRetinaNet()
    detector.setModelPath(os.path.join(execution_path,"models/resnet50_coco_best_v2.0.1.h5"))

detector.loadModel(detection_speed="fast")


cap = cv2.VideoCapture(os.path.join(execution_path, "data/{}.mp4".format(args["video"])))

ret, frame1 = cap.read()
ret, frame2 = cap.read()

def drawPerson(frame, box):
    # Draw a bounding box.
    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255, 178, 50), 2)
    return frame


print("width : {}".format(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
print("height : {}".format(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))


frame_num = 0
while cap.isOpened():
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
        
        frame_detected = frame1#[y:y+h, x:x+w]#cv2.crop_img(frame1, (x, y), (x+w, y+h), (0, 0, 255), 2))
        # cv2.imshow("detected image", frame_detected)
        # predictions, probabilities = prediction.predictImage(frame_detected, result_count=10, input_type="array")
        custom_objects = detector.CustomObjects(person=True, motorcycle=True)

        # detections = detector.detectCustomObjectsFromImage(custom_objects=custom_objects, input_image=frame_detected, input_type="array",output_image_path=os.path.join(execution_path , "detected/detected_img_{}.jpg".format(frame_num)),minimum_percentage_probability=60)

        detected_image_array, detections = detector.detectCustomObjectsFromImage(output_type="array", custom_objects=custom_objects, input_image=frame_detected, input_type="array")#,minimum_percentage_probability=30)


        for eachObject in detections:
            print(eachObject["name"] , " : ", eachObject["percentage_probability"], " : ", eachObject["box_points"] )
            print("--------------------------------")
            if eachObject["name"] == "person" and eachObject["percentage_probability"] > 30: 
                cv2.imshow("person", drawPerson(detected_image_array, eachObject["box_points"]))

            
        frame_num= frame_num+1
        # for predict in predictions:
        #     print(predict)


        # cv2.imshow("crop image", frame_detected)
        # cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 0, 255), 2)

        # crop_img = frame1[y:y+h, x:x+w]
        


        # cv2.putText(frame1, "Motion Detected !!!", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
         

    # cv2.drawContours(frame1, contours, -1, (0, 255, 0), 2)

    cv2.imshow("survilence", frame1)

    frame1 = frame2
    ret, frame2 = cap.read()

    if cv2.waitKey(40)==27:
        break

cv2.destroyAllWindows()
cap.release()


    
    # label = '%.2f' % conf
        
    # Get the label for the class name and its confidence
    # if classes:
    #     assert(classId < len(classes))
    #     label = '%s:%s' % (classes[classId], label)

    #Display the label at the top of the bounding box
    # labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    # top = max(top, labelSize[1])
    # cv.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine), (255, 255, 255), cv.FILLED)
    # cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)




# def detectObjeInFrame(framePath):
#     detector.detectObjectsFromVideo(input_file_path=os.path.join(execution_path, "data/thief_video.mp4"),
#                             output_file_path=os.path.join(execution_path, "data/thief_video_processed")
#                             , frames_per_second=20, log_progress=True)


