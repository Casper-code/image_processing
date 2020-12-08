#!/usr/bin/python

import numpy as np
import argparse
import time
import cv2 as cv
import os

def runYOLODetection(args):
    # load my fish class labels that my YOLO model was trained on
    labelsPath = os.path.sep.join([args["yolo"], "casper.names"])
    #labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
    LABELS = open(labelsPath).read().strip().split("\n")

    # initialize a list of colors to represent each possible class label
    np.random.seed(0)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
        dtype="uint8")
    print(COLORS)
    #COLORS = np.array([255, 0, 0], dtype="uint8")

    # derive the paths to the YOLO weights and model configuration
    weightsPath = os.path.sep.join([args["yolo"], "casper_final.weights"])
    configPath = os.path.sep.join([args["yolo"], "casper.cfg"])
    #weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
    #configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

    # load my YOLO object detector trained on my fish dataset (1 class)
    print("[INFO] loading YOLO from disk ...")
    net = cv.dnn.readNetFromDarknet(configPath, weightsPath)

    # load input image and grab its spatial dimensions
    image = cv.imread(args["image"])
    (H, W) = image.shape[:2]
    print(args["image"])
    for lista in os.listdir('./images'):
        print(lista)
    
    filename='test.jpg'
    
    # determine only the *output* layer names that we need from YOLO
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # construct a blob from the input image and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes and
    # associated probabilities
    # NOTE: (608, 608) is my YOLO input image size. However, using 
    # (416, 416) results in much accutate result. Pretty interesting.
    blob = cv.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
        swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()

    # show execution time information of YOLO
    print("[INFO] YOLO took {:.6f} seconds.".format(end - start))

    # initialize out lists of detected bounding boxes, confidences, and
    # class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []

    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            
            # filter out weak predictions by ensuring the detected
            # probability is greater then the minimum probability
            if confidence > args["confidence"]:
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top and
                # left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update out list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # apply non-maxima suppression to suppress weark and overlapping bounding
    # boxes
    idxs = cv.dnn.NMSBoxes(boxes, confidences, args["confidence"],
        args["threshold"])
    i=0
    t=0
    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # draw a bounding box rectangle and label on the image
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv.rectangle(image, (x, y), (x + w, y + h), color, 2)
            croped=image[y:y+h,x:x+w]
            if LABELS[classIDs[i]].startswith('image'):
                i+=1
                if not os.path.exists('CASPER/Images'):
                    os.makedirs('CASPER/Images')
                absolute_path_img= os.path.join(os.getcwd(),'CASPER/Images/',str(LABELS[classIDs[i]]) + '_' + str(i) + '_'+filename)
                print(absolute_path_img)
                cv.imwrite(absolute_path_img, croped)
            if LABELS[classIDs[i]].startswith('text'):
                t+=1
                if not os.path.exists('CASPER/Text'):
                    os.makedirs('CASPER/Text')
                absolute_path_txt= os.path.join(os.getcwd(),'CASPER/Text/',str(LABELS[classIDs[i]]) + '_' + str(t) + '_'+filename)
                print(absolute_path_txt)
                cv.imwrite(absolute_path_txt, croped)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            cv.putText(image, text, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX,
                0.5, color, 2)

    return image

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True,
        help="path to input image")
    ap.add_argument("-y", "--yolo", required=True,
        help="base path to YOLO directory")
    ap.add_argument("-c", "--confidence", type=float, default=0.25,
        help="minimum probability to filter weak detections")
    ap.add_argument("-t", "--threshold", type=float, default=0.45,
        help="threshold when applying non-maxima suppression")
    args = vars(ap.parse_args())

    image = runYOLODetection(args)

    # show the output image
    #cv.namedWindow("Image", cv.WINDOW_NORMAL)
    cv.resizeWindow("image", 1920, 1080)
    #cv.imshow("Image", image)
    cv.imwrite("predictions.jpg", image)
    cv.waitKey(0)
