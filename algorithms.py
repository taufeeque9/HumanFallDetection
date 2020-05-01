import io
import numpy as np
import openpifpaf
import PIL
import torch
import cv2
import argparse
import csv
import logging
import base64
import time
import multiprocessing

from typing import List
from visual import CocoPart, write_on_image, visualise
from processor import Processor
from helpers import pop_and_add
from default_params import *
from writer import write_to_csv
def alg2(queue,consecutive_frames=DEFAULT_CONSEC_FRAMES):
    counter2 = 0
    while True:
        if queue.empty():
        	continue

        M = queue.get()
        if M is None:
        	print("We are done")
        	break



        counter2+=1
        print(counter2)

        time.sleep(0.15)

def extract_keypoints(queue,args,consecutive_frames=DEFAULT_CONSEC_FRAMES):
    print('main started')
    M=[]
    if args.video is None:
        logging.debug('Video source: webcam')
        cam = cv2.VideoCapture(0)
    else:
        logging.debug(f'Video source: {args.video}')
        cam = cv2.VideoCapture(args.video)

    # Setup CSV file
    with open(args.csv_path, mode='w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"',
                                quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(['frame_no',
                             'nose.x', 'nose.y', 'nose.prob',
                             'l.eye.x', 'l.eye.y', 'l.eye.prob',
                             'r.eye.x', 'r.eye.y', 'r.eye.prob',
                             'l.ear.x', 'l.ear.y', 'l.ear.prob',
                             'r.ear.x', 'r.ear.y', 'r.ear.prob',
                             'l.shoulder.x', 'l.shoulder.y', 'l.shoulder.prob',
                             'r.shoulder.x', 'r.shoulder.y', 'r.shoulder.prob',
                             'l.elbow.x', 'l.elbow.y', 'l.elbow.prob',
                             'r.elbow.x', 'r.elbow.y', 'r.elbow.prob',
                             'l.wrist.x', 'l.wrist.y', 'l.wrist.prob',
                             'r.wrist.x', 'r.wrist.y', 'r.wrist.prob',
                             'l.hip.x', 'l.hip.y', 'l.hip.prob',
                             'r.hip.x', 'r.hip.y', 'r.hip.prob',
                             'l.knee.x', 'l.knee.y', 'l.knee.prob',
                             'r.knee.x', 'r.knee.y', 'r.knee.prob',
                             'l.ankle.x', 'l.ankle.y', 'l.ankle.prob',
                             'r.ankle.x', 'r.ankle.y', 'r.ankle.prob',
                             ])

    ret_val, img = cam.read()

    # Resize the video
    if args.resize is None:
        height, width = img.shape[:2]
    else:
        width, height = [int(dim) for dim in args.resize.split('x')]

    # width_height = (width, height)
    width_height = (int(width * args.resolution // 16) * 16,
                    int(height * args.resolution // 16) * 16)
    logging.debug(f'Target width and height = {width_height}')
    processor_singleton = Processor(width_height, args)

    # output_video = None

    frame = 0
    fps = 0
    t0 = time.time()
    cv2.namedWindow('Detected Pose')

    while time.time() - t0 < 10:

        ret_val, img = cam.read()
        if img is None:
            print('no more images captured')
            queue.put(None)
            break

        if cv2.waitKey(1) == 27 or cv2.getWindowProperty('Detected Pose', cv2.WND_PROP_VISIBLE) < 1:
            queue.put(None)
            break

        img = cv2.resize(img, (width, height))
        frame += 1

        keypoint_sets, scores, width_height = processor_singleton.single_image(
            b64image=base64.b64encode(cv2.imencode('.jpg', img)[1]).decode('UTF-8')
        )
        keypoint_sets = [{
            'coordinates': keypoints.tolist(),
            'detection_id': i,
            'score': score,
            'width_height': width_height,
        } for i, (keypoints, score) in enumerate(zip(keypoint_sets, scores))]

        if(len(M) < consecutive_frames):
            M.append(keypoint_sets)
        else:
            pop_and_add(M, keypoint_sets)
            queue.put(M)

        img = visualise(img=img, keypoint_sets=keypoint_sets, width=width, height=height, vis_keypoints=args.joints,
                        vis_skeleton=args.skeleton)

        write_to_csv(frame_number=frame, humans=keypoint_sets,
                          width=width, height=height, csv_fp=args.csv_path)

        img = write_on_image(
            img=img, text=f"Avg FPS {frame//(time.time()-t0)}", color=[0, 0, 0])

        # print(f"...............{cv2.getWindowProperty('Detected Pose', 1)}................")

        cv2.imshow('Detected Pose', img)

    queue.put(None)