import cv2
import csv
import logging
import base64
import time
import matplotlib.pyplot as plt
from visual import write_on_image, visualise
from processor import Processor
from helpers import pop_and_add
from default_params import *
from writer import write_to_csv
from inv_pendulum import *


def alg2(queue, consecutive_frames=DEFAULT_CONSEC_FRAMES):
    re = []
    x = []
    plt.show()
    axes = plt.gca()
    line, = axes.plot(x, re, 'r-')
    while True:
        if queue.qsize() > 0:
            curr = queue.get()
            if is_valid(curr[-1][0]['coordinates']):
                if is_valid(curr[-2][0]['coordinates']):
                    re.append(get_rot_energy(
                        curr[-2][0]['coordinates'], curr[-1][0]['coordinates']))
                elif is_valid(curr[-3][0]['coordinates']):
                    re.append(get_rot_energy(
                        curr[-3][0]['coordinates'], curr[-1][0]['coordinates']))
                elif is_valid(curr[-4][0]['coordinates']):
                    re.append(get_rot_energy(
                        curr[-4][0]['coordinates'], curr[-1][0]['coordinates']))
                else:
                    re.append(0)
            else:
                re.append(0)
        x = range(len(re))
        line.set_xdata(x)
        line.set_ydata(re)
        plt.draw()
        plt.pause(0.01)

    plt.show()


def extract_keypoints(queue, args, consecutive_frames=DEFAULT_CONSEC_FRAMES):
    print('main started')
    M = []
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

        if args.coco_points:
            keypoint_sets = [{
                'coordinates': keypoints.tolist(),
                'detection_id': i,
                'score': score,
                'width_height': width_height,
            } for i, (keypoints, score) in enumerate(zip(keypoint_sets, scores))]
        else:
            keypoint_sets = [{
                'coordinates': get_kp(keypoints.tolist()),
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
                        vis_skeleton=args.skeleton, CocoPointsOn=args.coco_points)

        # write_to_csv(frame_number=frame, humans=keypoint_sets,
        #              width=width, height=height, csv_fp=args.csv_path)

        img = write_on_image(
            img=img, text=f"Avg FPS {frame//(time.time()-t0)}", color=[0, 0, 0])

        # print(f"...............{cv2.getWindowProperty('Detected Pose', 1)}................")

        cv2.imshow('Detected Pose', img)

    queue.put(None)
