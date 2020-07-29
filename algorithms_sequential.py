import cv2
import logging
import base64
import time
import numpy as np
import matplotlib.pyplot as plt
from visual import write_on_image, visualise, activity_dict, visualise_tracking
from processor import Processor
from helpers import pop_and_add, last_ip, dist, move_figure
from default_params import *
from inv_pendulum import *
import re
import pandas as pd
from scipy.signal import savgol_filter, lfilter
from livetest import LSTMModel
import torch


def show_tracked_img(img_dict, ip_set, num_matched):
    img = img_dict["img"]
    tagged_df = img_dict["tagged_df"]
    keypoints_frame = [person[-1] for person in ip_set]
    img = visualise_tracking(img=img, keypoint_sets=keypoints_frame, width=img_dict["width"], height=img_dict["height"],
                             num_matched=num_matched, vis_keypoints=img_dict["vis_keypoints"], vis_skeleton=img_dict["vis_skeleton"],
                             CocoPointsOn=False)

    img = write_on_image(img=img, text=tagged_df["text"],
                         color=tagged_df["color"])
    return img


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


def remove_wrongly_matched(matched_1, matched_2):

    unmatched_idxs = []
    i = 0

    for ip1, ip2 in zip(matched_1, matched_2):
        # each of these is a set of the last t framses of each matched person
        correlation = cv2.compareHist(last_valid_hist(ip1)["up_hist"], last_valid_hist(ip2)["up_hist"], cv2.HISTCMP_CORREL)
        if correlation < 0.5*HIST_THRESH:
            unmatched_idxs.append(i)
        i += 1

    return unmatched_idxs


def match_unmatched(unmatched_1, unmatched_2, lstm_set1, lstm_set2, num_matched):

    new_matched_1 = []
    new_matched_2 = []
    new_lstm1 = []
    new_lstm2 = []
    final_pairs = [[], []]

    if not unmatched_1 or not unmatched_2:
        return final_pairs, new_matched_1, new_matched_2, new_lstm1, new_lstm2

    new_matched = 0
    correlation_matrix = - np.ones((len(unmatched_1), len(unmatched_2)))
    dist_matrix = np.zeros((len(unmatched_1), len(unmatched_2)))
    for i in range(len(unmatched_1)):
        for j in range(len(unmatched_2)):
            correlation_matrix[i][j] = cv2.compareHist(last_valid_hist(unmatched_1[i])["up_hist"],
                                                       last_valid_hist(unmatched_2[j])["up_hist"], cv2.HISTCMP_CORREL)
            dist_matrix[i][j] = np.sum(np.absolute(last_valid_hist(unmatched_1[i])["up_hist"]-last_valid_hist(unmatched_2[j])["up_hist"]))

    freelist_1 = [i for i in range(len(unmatched_1))]
    pair_21 = [-1]*len(unmatched_2)
    unmatched_1_preferences = np.argsort(-correlation_matrix, axis=1)
    print("cor", correlation_matrix, sep="\n")
    print("unmatched_1", unmatched_1_preferences, sep="\n")
    unmatched_indexes1 = [0]*len(unmatched_1)
    finish_array = [False]*len(unmatched_1)
    while freelist_1:
        um1_idx = freelist_1[-1]
        if finish_array[um1_idx] == True:
            freelist_1.pop()
            continue
        next_unasked_2 = unmatched_1_preferences[um1_idx][unmatched_indexes1[um1_idx]]
        if pair_21[next_unasked_2] == -1:
            pair_21[next_unasked_2] = um1_idx
            freelist_1.pop()
        else:
            curr_paired_2 = pair_21[next_unasked_2]
            if correlation_matrix[curr_paired_2][next_unasked_2] < correlation_matrix[um1_idx][next_unasked_2]:
                pair_21[next_unasked_2] = um1_idx
                freelist_1.pop()
                if not finish_array[curr_paired_2]:
                    freelist_1.append(curr_paired_2)

        unmatched_indexes1[um1_idx] += 1
        if unmatched_indexes1[um1_idx] == len(unmatched_2):
            finish_array[um1_idx] = True

    for j, i in enumerate(pair_21):
        if correlation_matrix[i][j] > HIST_THRESH:
            final_pairs[0].append(i+num_matched)
            final_pairs[1].append(j+num_matched)
            new_matched_1.append(unmatched_1[i])
            new_matched_2.append(unmatched_2[j])
            new_lstm1.append(lstm_set1[i])
            new_lstm2.append(lstm_set2[j])

    print("finalpairs", final_pairs, sep="\n")

    return final_pairs, new_matched_1, new_matched_2, new_lstm1, new_lstm2


def alg2_sequential(queue1, queue2, args1, args2, consecutive_frames=DEFAULT_CONSEC_FRAMES, feature_q1=None, feature_q2=None):
    model = LSTMModel(h_RNN=32, h_RNN_layers=2, drop_p=0.2, num_classes=7)
    model.load_state_dict(torch.load('lstm.sav'))
    model.eval()
    t0 = time.time()
    feature_plotter1 = [[], [], [], [], []]
    feature_plotter2 = [[], [], [], [], []]
    ip_set1, ip_set2 = [], []
    lstm_set1, lstm_set2 = [], []
    max_length_mat = 300
    num_matched = 0
    if not args1.plot_graph:
        max_length_mat = consecutive_frames
    else:
        f, ax = plt.subplots()
        move_figure(f, 800, 100)

    cv2.namedWindow(args1.video)
    cv2.namedWindow(args2.video)
    while True:
        if not queue1.empty() and not queue2.empty():
            dict_frame1 = queue1.get()
            dict_frame2 = queue2.get()
            if dict_frame1 is None or dict_frame2 is None:
                break

            if cv2.waitKey(1) == 27 or cv2.getWindowProperty(args1.video, cv2.WND_PROP_VISIBLE) < 1:
                break
            if cv2.waitKey(1) == 27 or cv2.getWindowProperty(args2.video, cv2.WND_PROP_VISIBLE) < 1:
                break
            kp_frame1 = dict_frame1["keypoint_sets"]
            kp_frame2 = dict_frame2["keypoint_sets"]
            num_matched, new_num, indxs_unmatched1 = match_ip(ip_set1, kp_frame1, lstm_set1, num_matched, max_length_mat)
            assert(new_num == len(ip_set1))
            for i in sorted(indxs_unmatched1, reverse=True):
                elem = ip_set2[i]
                ip_set2.pop(i)
                ip_set2.append(elem)
                elem_lstm = lstm_set2[i]
                lstm_set2.pop(i)
                lstm_set2.append(elem_lstm)
            num_matched, new_num, indxs_unmatched2 = match_ip(ip_set2, kp_frame2, lstm_set2, num_matched, max_length_mat)

            for i in sorted(indxs_unmatched2, reverse=True):
                elem = ip_set1[i]
                ip_set1.pop(i)
                ip_set1.append(elem)
                elem_lstm = lstm_set1[i]
                lstm_set1.pop(i)
                lstm_set1.append(elem_lstm)

            matched_1 = ip_set1[:num_matched]
            matched_2 = ip_set2[:num_matched]

            unmatch_previous = remove_wrongly_matched(matched_1, matched_2)
            if unmatch_previous:
                print(unmatch_previous)

            for i in sorted(unmatch_previous, reverse=True):
                elem1 = ip_set1[i]
                elem2 = ip_set2[i]
                ip_set1.pop(i)
                ip_set2.pop(i)
                ip_set1.append(elem1)
                ip_set2.append(elem2)
                elem_lstm1 = lstm_set1[i]
                lstm_set1.pop(i)
                lstm_set1.append(elem_lstm1)
                elem_lstm2 = lstm_set2[i]
                lstm_set2.pop(i)
                lstm_set2.append(elem_lstm2)
                num_matched -= 1

            unmatched_1 = ip_set1[num_matched:]
            unmatched_2 = ip_set2[num_matched:]

            new_pairs, new_matched1, new_matched2, new_lstm1, new_lstm2 = match_unmatched(
                unmatched_1, unmatched_2, lstm_set1, lstm_set2, num_matched)

            new_p1 = new_pairs[0]
            new_p2 = new_pairs[1]

            for i in sorted(new_p1, reverse=True):
                ip_set1.pop(i)
                lstm_set1.pop(i)
            for i in sorted(new_p2, reverse=True):
                ip_set2.pop(i)
                lstm_set2.pop(i)

            ip_set1 = ip_set1[:num_matched] + new_matched1 + ip_set1[num_matched:]
            ip_set2 = ip_set2[:num_matched] + new_matched2 + ip_set2[num_matched:]
            lstm_set1 = lstm_set1[:num_matched] + new_lstm1 + lstm_set1[num_matched:]
            lstm_set2 = lstm_set2[:num_matched] + new_lstm2 + lstm_set2[num_matched:]
            # remember to match the energy matrices also

            num_matched = num_matched + len(new_matched1)

            # get features now

            valid1_idxs, prediction1 = get_all_features(ip_set1, lstm_set1, model)
            valid2_idxs, prediction2 = get_all_features(ip_set2, lstm_set2, model)
            dict_frame1["tagged_df"]["text"] += f"Pred: {activity_dict[prediction1+5]}"
            dict_frame2["tagged_df"]["text"] += f"Pred: {activity_dict[prediction2+5]}"
            img1 = show_tracked_img(dict_frame1, ip_set1, num_matched)
            img2 = show_tracked_img(dict_frame2, ip_set2, num_matched)
            # print(img1.shape)
            cv2.imshow(args1.video, img1)
            cv2.imshow(args2.video, img2)

            assert(len(lstm_set1) == len(ip_set1))
            assert(len(lstm_set2) == len(ip_set2))

            DEBUG = False
            for ip_set, feature_plotter in zip([ip_set1, ip_set2], [feature_plotter1, feature_plotter2]):
                for cnt in range(len(FEATURE_LIST)):
                    plt_f = FEATURE_LIST[cnt]
                    if ip_set and ip_set[0] is not None and ip_set[0][-1] is not None and plt_f in ip_set[0][-1]["features"]:
                        # print(ip_set[0][-1]["features"])
                        feature_plotter[cnt].append(ip_set[0][-1]["features"][plt_f])

                    else:
                        # print("None")
                        feature_plotter[cnt].append(0)
                DEBUG = True

    cv2.destroyAllWindows()

    for i, feature_arr in enumerate(feature_plotter1):
        plt.clf()
        x = np.linspace(1, len(feature_arr), len(feature_arr))
        axes = plt.gca()
        filter_array = feature_arr
        line, = axes.plot(x, filter_array, 'r-')
        plt.ylabel(FEATURE_LIST[i])
        # #plt.savefig(f'{args1.video}_{FEATURE_LIST[i]}_filter.png')
        plt.pause(1e-7)

    for i, feature_arr in enumerate(feature_plotter2):
        plt.clf()
        x = np.linspace(1, len(feature_arr), len(feature_arr))
        axes = plt.gca()
        filter_array = feature_arr
        line, = axes.plot(x, filter_array, 'r-')
        plt.ylabel(FEATURE_LIST[i])
        # plt.savefig(f'{args2.video}_{FEATURE_LIST[i]}_filter.png')
        plt.pause(1e-7)
        # if len(re_matrix1[0]) > 0:
        #     print(np.linalg.norm(ip_set1[0][-1][0]['B']-ip_set1[0][-1][0]['H']))

    print("P2 Over")
    return


def get_all_features(ip_set, lstm_set, model):
    valid_idxs = []
    invalid_idxs = []

    for i, ips in enumerate(ip_set):
        # ip set for a particular person
        last1 = None
        last2 = None
        for j in range(-2, -1*DEFAULT_CONSEC_FRAMES - 1, -1):
            if ips[j] is not None:
                if last1 is None:
                    last1 = j
                elif last2 is None:
                    last2 = j
        if ips[-1] is None:
            invalid_idxs.append(i)
            # continue
        else:
            ips[-1]["features"] = {}
            # get re, gf, angle, bounding box ratio, ratio derivative

            ips[-1]["features"]["ratio_bbox"] = FEATURE_SCALAR["ratio_bbox"]*get_ratio_bbox(ips[-1])

            body_vector = ips[-1]["keypoints"]["N"] - ips[-1]["keypoints"]["B"]
            ips[-1]["features"]["angle_vertical"] = FEATURE_SCALAR["angle_vertical"]*get_angle_vertical(body_vector)
            ips[-1]["features"]["log_angle"] = FEATURE_SCALAR["log_angle"]*np.log(1 + np.abs(ips[-1]["features"]["angle_vertical"]))

            if last1 is None:
                invalid_idxs.append(i)
                # continue
            else:
                ips[-1]["features"]["re"] = FEATURE_SCALAR["re"]*get_rot_energy(ips[last1], ips[-1])
                ips[-1]["features"]["ratio_derivative"] = FEATURE_SCALAR["ratio_derivative"]*get_ratio_derivative(ips[last1], ips[-1])

                if last2 is None:
                    invalid_idxs.append(i)
                    # continue
                else:
                    ips[-1]["features"]["gf"] = get_gf(ips[last2], ips[last1], ips[-1])
                    valid_idxs.append(i)

        xdata = []
        if ips[-1] is None:
            for feat in FEATURE_LIST[:FRAME_FEATURES]:
                xdata.append(ips[last1]["features"][feat])
            xdata += [0]*(len(FEATURE_LIST)-FRAME_FEATURES)
        else:
            for feat in FEATURE_LIST:
                if feat in ips[-1]["features"]:
                    xdata.append(ips[-1]["features"][feat])
                else:
                    xdata.append(0)

        xdata = torch.Tensor(xdata).view(-1, 1, 5)
        # what is ips[-2] is none
        outputs, lstm_set[i] = model(xdata, lstm_set[i])
        prediction = torch.max(outputs.data, 1)[1][0].item()

    return valid_idxs, prediction


def get_frame_features(ip_set, new_frame, re_matrix, gf_matrix, num_matched, max_length_mat=DEFAULT_CONSEC_FRAMES):

    match_ip(ip_set, new_frame, re_matrix, gf_matrix, max_length_mat)
    return
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
