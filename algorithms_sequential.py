import cv2
import logging
import base64
import time
import numpy as np
import matplotlib.pyplot as plt
from visual import write_on_image, visualise, activity_dict
from processor import Processor
from helpers import pop_and_add, last_ip, dist, move_figure
from default_params import *
from inv_pendulum import *
import re
import pandas as pd


def get_source(args, i):
    tagged_df = None
    if args.video is None:
        logging.debug(f'Video source {i}: webcam')
        cam = cv2.VideoCapture(0)
    else:
        logging.debug(f'Video source: {args.video}')
        cam = cv2.VideoCapture(args.video)
        vid = [int(s) for s in re.findall(r'\d+', args.video)]
        if len(vid) == 5:
            tagged_df = pd.read_csv("dataset/CompleteDataSet.csv", usecols=[
                                    "TimeStamps", "Subject", "Activity", "Trial", "Tag"], skipinitialspace=True)
            tagged_df = tagged_df.query(
                f'Subject == {vid[1]} & Activity == {vid[0]} & Trial == {vid[2]}')
    return cam, tagged_df


def resize(img, resize, resolution):

    if resize is None:
        height, width = img.shape[:2]
    else:
        width, height = [int(dim) for dim in resize.split('x')]
    width_height = (int(width * resolution // 16) * 16,
                    int(height * resolution // 16) * 16)
    return width, height, width_height


def process_vidframe(queue, args, cam, tagged_df, processor_singleton, width, height, frame, t0, output_video):
    ret_val, img = cam.read()
    if tagged_df is None:
        curr_time = time.time()
    else:
        curr_time = tagged_df.iloc[frame-1]['TimeStamps'][11:]
        curr_time = sum(x * float(t) for x, t in zip([3600, 60, 1], curr_time.split(":")))

    if img is None:
        print('no more images captured')
        queue.put(None)
        print(args.video, curr_time, sep=" ")
        return False

    if cv2.waitKey(1) == 27 or cv2.getWindowProperty(args.video, cv2.WND_PROP_VISIBLE) < 1:
        queue.put(None)
        return False

    img = cv2.resize(img, (width, height))

    keypoint_sets, scores, width_height = processor_singleton.single_image(
        b64image=base64.b64encode(cv2.imencode('.jpg', img)[1]).decode('UTF-8')
    )

    if args.coco_points:
        keypoint_sets = [keypoints.tolist() for keypoints in keypoint_sets]
    else:
        keypoint_sets = [(get_kp(keypoints.tolist()), curr_time) for keypoints in keypoint_sets]

    queue.put(keypoint_sets)
    img = visualise(img=img, keypoint_sets=keypoint_sets, width=width, height=height, vis_keypoints=args.joints,
                    vis_skeleton=args.skeleton, CocoPointsOn=args.coco_points)

    if tagged_df is None:
        img = write_on_image(
            img=img, text=f"Avg FPS: {frame//(time.time()-t0)}, Frame: {frame}", color=[0, 0, 0])
    else:
        img = write_on_image(img=img,
                             text=f"Avg FPS: {frame//(time.time()-t0)}, Frame: {frame}, Tag: {activity_dict[tagged_df.iloc[frame-1]['Tag']]}",
                             color=[0, 0, 0])

    if output_video is None:
        if args.save_output:
            output_video = cv2.VideoWriter(filename=args.out_path, fourcc=cv2.VideoWriter_fourcc(*'MP42'),
                                           fps=args.fps, frameSize=img.shape[:2][::-1])
            logging.debug(
                f'Saving the output video at {args.out_path} with {args.fps} frames per seconds')
        else:
            output_video = None
            logging.debug(f'Not saving the output video')
    else:
        output_video.write(img)
    cv2.imshow(args.video, img)

    return True


def extract_keypoints_sequential(queue1, queue2, args1, args2, consecutive_frames=DEFAULT_CONSEC_FRAMES):
    print('main started')

    cam1, tagged_df1 = get_source(args1, 1)
    cam2, tagged_df2 = get_source(args2, 2)
    ret_val1, img1 = cam1.read()
    ret_val2, img2 = cam2.read()
    # Resize the video
    width1, height1, width_height1 = resize(img1, args1.resize, args2.resolution)
    width2, height2, width_height2 = resize(img2, args2.resize, args2.resolution)

    # width_height = (width, height)
    logging.debug(f'Target width and height 1 = {width_height1}')
    processor_singleton1 = Processor(width_height1, args1)
    logging.debug(f'Target width and height 1 = {width_height2}')
    processor_singleton2 = Processor(width_height2, args2)
    output_video1 = None
    output_video2 = None
    frame = 0
    fps = 0
    t0 = time.time()
    cv2.namedWindow(args1.video)
    cv2.namedWindow(args2.video)
    max_time = 30

    while time.time() - t0 < max_time:
        # print(args.video,self_counter.value,other_counter.value,sep=" ")
        result1 = process_vidframe(queue1, args1, cam1, tagged_df1,
                                   processor_singleton1, width1, height1, frame, t0, output_video1)
        result2 = process_vidframe(queue2, args2, cam2, tagged_df2,
                                   processor_singleton2, width2, height2, frame, t0, output_video2)
        if not result1 or not result2:
            break
        frame += 1
    cv2.destroyAllWindows()
    queue1.put(None)
    queue2.put(None)
    print(f"Frames in {max_time}secs: {frame}")
    return


def alg2_sequential(queue1, queue2, plot_graph, consecutive_frames=DEFAULT_CONSEC_FRAMES, feature_q1=None, feature_q2=None):
    t0 = time.time()
    re_matrix1, re_matrix2 = [], []
    gf_matrix1, gf_matrix2 = [], []
    ip_set1, ip_set2 = [], []
    max_length_mat = 300
    if not plot_graph:
        max_length_mat = consecutive_frames
    else:
        f, ax = plt.subplots()
        move_figure(f, 800, 100)

    ip_set1, ip_set2 = [], []
    while True:
        if not queue1.empty() and not queue2.empty():
            new_frame1 = queue1.get()
            new_frame2 = queue2.get()
            if new_frame1 is None or new_frame2 is None:
                break
            get_frame_features(ip_set1, new_frame1, re_matrix1, gf_matrix1, max_length_mat)
            get_frame_features(ip_set2, new_frame2, re_matrix2, gf_matrix2, max_length_mat)

            # if len(re_matrix1[0]) > 0:
            #     print(np.linalg.norm(ip_set1[0][-1][0]['B']-ip_set1[0][-1][0]['H']))
    if feature_q1 is not None:
        feature_q1.put(re_matrix1)
        feature_q1.put(gf_matrix1)
    if feature_q2 is not None:
        feature_q2.put(re_matrix2)
        feature_q2.put(gf_matrix2)

    print("P2 Over")
    return


def get_frame_features(ip_set, new_frame, re_matrix, gf_matrix, max_length_mat=DEFAULT_CONSEC_FRAMES):

    match_ip(ip_set, new_frame, re_matrix, gf_matrix, max_length_mat)
    for i in range(len(ip_set)):
        if ip_set[i][-1] is not None:
            if ip_set[i][-2] is not None:
                pop_and_add(re_matrix[i], get_rot_energy(
                            ip_set[i][-2], ip_set[i][-1]), max_length_mat)
            elif ip_set[i][-3] is not None:
                pop_and_add(re_matrix[i], get_rot_energy(
                            ip_set[i][-3], ip_set[i][-1]), max_length_mat)
            elif ip_set[i][-4] is not None:
                pop_and_add(re_matrix[i], get_rot_energy(
                            ip_set[i][-4], ip_set[i][-1]), max_length_mat)
            else:
                pop_and_add(re_matrix[i], 0, max_length_mat)
        else:
            pop_and_add(re_matrix[i], 0, max_length_mat)

    for i in range(len(ip_set)):
        if ip_set[i][-1] is not None:
            last1 = None
            last2 = None
            for j in [-2, -3, -4, -5]:
                if ip_set[i][j] is not None:
                    if last1 is None:
                        last1 = j
                    elif last2 is None:
                        last2 = j

            if last2 is None:
                pop_and_add(gf_matrix[i], 0, max_length_mat)
                continue

            pop_and_add(gf_matrix[i], get_gf(ip_set[i][last2], ip_set[i][last1],
                                             ip_set[i][-1]), max_length_mat)

        else:

            pop_and_add(gf_matrix[i], 0, max_length_mat)

    return
