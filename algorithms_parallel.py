import cv2
import logging
import base64
import time
import numpy as np
import matplotlib.pyplot as plt
from visual import write_on_image, visualise, activity_dict
from processor import Processor
from helpers import pop_and_add, move_figure, get_hist
from default_params import *
from inv_pendulum import *
import re
import pandas as pd


def get_source(args):
    tagged_df = None
    if args.video is None:
        # logging.debug(f'Video source {i}: webcam')
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
    # Resize the video
    if resize is None:
        height, width = img.shape[:2]
    else:
        width, height = [int(dim) for dim in resize.split('x')]
    width_height = (int(width * resolution // 16) * 16,
                    int(height * resolution // 16) * 16)
    return width, height, width_height


def extract_keypoints_parallel(queue, args, self_counter, other_counter, consecutive_frames=DEFAULT_CONSEC_FRAMES):
    print('main started')

    cam, tagged_df = get_source(args)

    ret_val, img = cam.read()

    width, height, width_height = resize(img, args.resize, args.resolution)
    logging.debug(f'Target width and height = {width_height}')
    processor_singleton = Processor(width_height, args)

    output_video = None

    frame = 0
    fps = 0
    t0 = time.time()
    # cv2.namedWindow(args.video)
    max_time = 1000
    while time.time() - t0 < max_time:
        # print(args.video,self_counter.value,other_counter.value,sep=" ")
        if args.num_cams == 2 and (self_counter.value > other_counter.value):
            continue

        ret_val, img = cam.read()
        frame += 1
        self_counter.value += 1
        if tagged_df is None:
            curr_time = time.time()
        else:
            curr_time = tagged_df.iloc[frame-1]['TimeStamps'][11:]
            curr_time = sum(x * float(t) for x, t in zip([3600, 60, 1], curr_time.split(":")))

        if img is None:
            print('no more images captured')
            queue.put(None)
            print(args.video, curr_time, sep=" ")
            break

        # if cv2.waitKey(1) == 27 or cv2.getWindowProperty(args.video, cv2.WND_PROP_VISIBLE) < 1:
        #     queue.put(None)
        #     break

        img = cv2.resize(img, (width, height))
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        keypoint_sets, bb_list, width_height = processor_singleton.single_image(img)
        assert bb_list is None or (type(bb_list) == list)
        if bb_list:
            assert type(bb_list[0]) == tuple
            assert type(bb_list[0][0]) == tuple
        # assume bb_list is a of the form [(x1,y1),(x2,y2)),etc.]

        if args.coco_points:
            keypoint_sets = [keypoints.tolist() for keypoints in keypoint_sets]
        else:
            anns = [get_kp(keypoints.tolist()) for keypoints in keypoint_sets]
            ubboxes = [(np.asarray([width, height])*np.asarray(ann[1])).astype('int32')
                       for ann in anns]
            lbboxes = [(np.asarray([width, height])*np.asarray(ann[2])).astype('int32')
                       for ann in anns]
            bbox_list = [(np.asarray([width, height])*np.asarray(box)).astype('int32') for box in bb_list]
            uhist_list = [get_hist(hsv_img, bbox) for bbox in ubboxes]
            lhist_list = [get_hist(img, bbox) for bbox in lbboxes]
            keypoint_sets = [{"keypoints": keyp[0], "up_hist":uh, "lo_hist":lh, "time":curr_time, "box":box}
                             for keyp, uh, lh, box in zip(anns, uhist_list, lhist_list, bbox_list)]

            cv2.polylines(img, ubboxes, True, (255, 0, 0), 2)
            cv2.polylines(img, lbboxes, True, (0, 255, 0), 2)
            for box in bbox_list:
                cv2.rectangle(img, tuple(box[0]), tuple(box[1]), ((0, 0, 255)), 2)

        # img = visualise(img=img, keypoint_sets=keypoint_sets, width=width, height=height, vis_keypoints=args.joints,
        #                vis_skeleton=args.skeleton, CocoPointsOn=args.coco_points)

        # if tagged_df is None:
        #     img = write_on_image(
        #         img=img, text=f"Avg FPS: {frame//(time.time()-t0)}, Frame: {frame}", color=[0, 0, 0])
        # else:
        #     img = write_on_image(img=img,
        #                          text=f"Avg FPS: {frame//(time.time()-t0)}, Frame: {frame}, Tag: {activity_dict[tagged_df.iloc[frame-1]['Tag']]}",
        #                          color=[0, 0, 0])

        # if output_video is None:
        #     if args.save_output:
        #         output_video = cv2.VideoWriter(filename=args.out_path, fourcc=cv2.VideoWriter_fourcc(*'MP42'),
        #                                        fps=args.fps, frameSize=img.shape[:2][::-1])
        #         logging.debug(
        #             f'Saving the output video at {args.out_path} with {args.fps} frames per seconds')
        #     else:
        #         output_video = None
        #         logging.debug(f'Not saving the output video')
        # else:
        #     output_video.write(img)
        dict_vis = {"img": img, "keypoint_sets": keypoint_sets, "width": width, "height": height, "vis_keypoints": args.joints,
                    "vis_skeleton": args.skeleton, "CocoPointsOn": args.coco_points,
                    "tagged_df": {"text": f"Avg FPS: {frame//(time.time()-t0)}, Frame: {frame}", "color": [0, 0, 0]}}
        queue.put(dict_vis)
        #cv2.imshow(args.video, img)

    print(f"Frames in {max_time}secs: {frame}")
    # cv2.destroyWindow(args.video)
    queue.put(None)

    return


def alg2_parallel(queue, plot_graph, consecutive_frames=DEFAULT_CONSEC_FRAMES, feature_q=None):
    re_matrix = []
    gf_matrix = []
    max_length_mat = 200
    if not plot_graph:
        max_length_mat = consecutive_frames
    else:
        f, ax = plt.subplots()
        move_figure(f, 800, 100)
    ip_set = []

    while True:
        if not queue.empty():
            new_frame = queue.get()
            if new_frame is None:
                break

            match_ip(ip_set, new_frame, re_matrix, gf_matrix, max_length_mat)
            len_ip = [len(k) for k in ip_set]
            len_re = [len(k) for k in re_matrix]
            # print("IP: ",len_ip)
            # print("RE: ",len_re)
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
        if not plot_graph:
            continue
        if feature_q is None and len(re_matrix) > 0:
            plt.clf()
            x = np.linspace(1, len(re_matrix[0]), len(re_matrix[0]))
            axes = plt.gca()
            line, = axes.plot(x, re_matrix[0], 'r-')
            plt.draw()
            plt.pause(1e-17)

    # print(ip_set)
    # print(re_matrix)
    if feature_q is not None:
        feature_q.put(re_matrix)
        feature_q.put(gf_matrix)
