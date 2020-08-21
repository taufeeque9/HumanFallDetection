import joblib
import cv2
import pandas as pd
from default_params import DEFAULT_CONSEC_FRAMES, FEATURE_LIST
from algorithms_sequential import get_all_features
from inv_pendulum import *
from queue import Queue
import time
import numpy as np
t0 = time.time()


def get_bb(kp_set, width, height, score=None):
    bb_list = []
    for i in range(kp_set.shape[0]):
        x = kp_set[i, :15, 0]
        y = kp_set[i, :15, 1]
        v = kp_set[i, :15, 2]
        assert np.any(v > 0)
        if not np.any(v > 0):
            return None

        # keypoint bounding box
        x1, x2 = np.min(x[v > 0]), np.max(x[v > 0])
        y1, y2 = np.min(y[v > 0]), np.max(y[v > 0])
        if x2 - x1 < 5.0/width:
            x1 -= 2.0/width
            x2 += 2.0/width
        if y2 - y1 < 5.0/height:
            y1 -= 2.0/height
            y2 += 2.0/height

        bb_list.append(((x1, y1), (x2, y2)))

    # ax.add_patch(
    #     matplotlib.patches.Rectangle(
    #         (x1, y1), x2s - x1, y2 - y1, fill=False, color=color))
    #
    # if score:
    #     ax.text(x1, y1, '{:.4f}'.format(score), fontsize=8, color=color)
    return bb_list


def alg2_sequential(queue1, plot_graph, consecutive_frames=DEFAULT_CONSEC_FRAMES, feature_q1=None, feature_q2=None):
    feature_plotter1 = [[], [], [], [], [], []]
    feature_plotter2 = [[], [], [], [], [], []]
    ip_set1, ip_set2 = [], []
    max_length_mat = 1000
    num_matched = 0
    if not plot_graph:
        max_length_mat = consecutive_frames

    # cv2.namedWindow(args1.video)
    # cv2.namedWindow(args2.video)
    while True:
        if not queue1.empty():
            dict_frame1 = queue1.get()
            if dict_frame1 is None:
                break

            # if cv2.waitKey(1) == 27 or cv2.getWindowProperty(args1.video, cv2.WND_PROP_VISIBLE) < 1:
            #     break
            # if cv2.waitKey(1) == 27 or cv2.getWindowProperty(args2.video, cv2.WND_PROP_VISIBLE) < 1:
            #     break
            kp_frame1 = dict_frame1["keypoint_sets"]
            # kp_frame2 = dict_frame2["keypoint_sets"]
            num_matched, new_num, indxs_unmatched1 = match_ip(ip_set1, kp_frame1, [], num_matched, max_length_mat)
            # assert(new_num == len(ip_set1))

            valid1_idxs = get_all_features(ip_set1, [], [])
            # valid2_idxs = get_all_features(ip_set2)
            cnt = 0
            continue
            for ip_set, feature_plotter in zip([ip_set1, ip_set2], [feature_plotter1, feature_plotter2]):
                plt_f = FEATURE_LIST[cnt]
                if ip_set and ip_set[0] is not None and ip_set[0][-1] is not None:
                    # print(ip_set[0][-1]["features"])
                    feature_plotter[cnt].append(ip_set[0][-1]["features"][plt_f])
                else:
                    # print("None")
                    feature_plotter[cnt].append(0)
                cnt += 1

    # cv2.destroyAllWindows()

    if feature_q1 is not None:
        feature_q1.put(ip_set1)
    # for i in range(feature_plotter1):
    #     plt.clf()
    #     x = np.linspace(1, len(feature_plotter1), len(feature_plotter1))
    #     axes = plt.gca()
    #     line, = axes.plot(x, feature_plotter1, 'r-')
    #     plt.draw()
    #     plt.pause(1e-1)
    #
    #     # if len(re_matrix1[0]) > 0:
    #     #     print(np.linalg.norm(ip_set1[0][-1][0]['B']-ip_set1[0][-1][0]['H']))
    #
    # print("P2 Over")
    # return


def alg1(trial, df):
    width, height = trial[-1]
    queue = Queue()
    f_q = Queue()
    for timeframe, keypoint_sets in enumerate(trial[:-1]):
        # print(time)
        bb_list = get_bb(np.asarray(keypoint_sets), width, height)
        anns = [get_kp(keypoints) for keypoints in keypoint_sets]
        # ubboxes = [(np.asarray([width, height])*np.asarray(ann[1])).astype('int32')
        #            for ann in anns]
        # lbboxes = [(np.asarray([width, height])*np.asarray(ann[2])).astype('int32')
        #            for ann in anns]
        bbox_list = [(np.asarray([width, height])*np.asarray(box)).astype('int32') for box in bb_list]
        # uhist_list = [get_hist(hsv_img, bbox) for bbox in ubboxes]
        # lhist_list = [get_hist(img, bbox) for bbox in lbboxes]
        curr_time = df.iloc[timeframe]['TimeStamps'][11:]
        curr_time = sum(x * float(t) for x, t in zip([3600, 60, 1], curr_time.split(":")))
        anns = [{"keypoints": keyp[0], "time":curr_time, "box":box}
                for keyp, box in zip(anns, bbox_list)]
        dict_vis = {"keypoint_sets": anns, "width": width, "height": height,
                    }
        #
        # cv2.polylines(img, ubboxes, True, (255, 0, 0), 2)
        # cv2.polylines(img, lbboxes, True, (0, 255, 0), 2)
        # for box in bbox_list:
        #     cv2.rectangle(img, tuple(box[0]), tuple(box[1]), ((0, 0, 255)), 2)

        queue.put(dict_vis)
    queue.put(None)
    alg2_sequential(queue1=queue, plot_graph=True, feature_q1=f_q)
    return f_q


def extract(sub_start, sub_end, csv_name):

    tagged_df = pd.read_csv('dataset/CompleteDataSet.csv', usecols=[
                            "TimeStamps", "Subject", "Activity", "Trial", "Tag"], skipinitialspace=True)
    final_data = []

    for sub in range(sub_start, sub_end):
        print(sub)
        t0 = time.time()
        for act in range(1, 12):

            # print(act)
            trials = joblib.load(f'cocokps/act{act}sub{sub}.kps')
            for trial_num, trial in enumerate(trials):
                df = tagged_df.query(
                    f'Subject == {sub} & Activity == {act} & Trial == {trial_num//2+1}')

                if trial_num % 2 == 1:
                    continue

                # if (act in [6, 7, 8, 10] and trial_num > 1) or (act != 8 and trial_num % 2 != 1):
                #     continue
                # if act == 8 and trial_num % 2 == 1:
                #     continue
                # if (act in [6, 7, 8, 10] and trial_num > 1):
                #     continue
                if act in [1, 2, 3, 4, 5, 9]:
                    for flip in [False, True]:
                        if flip:
                            for frame in trial[:-1]:
                                for kp in frame:
                                    for i, (x, _, p) in enumerate(kp):
                                        kp[i][0] = 1 - x if p > 0 else x
                        f_q = alg1(trial, df)
                        try:
                            ip_set = f_q.get()[0]
                        except:
                            continue
                        zero = 1000 - len(trial) + 1 - (DEFAULT_CONSEC_FRAMES - 1)
                        zero = max(0, zero)
                        for i in range(zero, 1000 - len(trial) + 1):
                            ip_set[i] = ip_set[1000 - len(trial) + 1]

                        last = len(ip_set)

                        prevprev = {feat: 0 for feat in FEATURE_LIST}
                        for i in range(zero+1+DEFAULT_CONSEC_FRAMES, last):
                            row = []
                            prev = {feat: prevprev[feat] for feat in FEATURE_LIST}

                            for feat in FEATURE_LIST:
                                for frame in ip_set[i+1-DEFAULT_CONSEC_FRAMES:i+1]:
                                    if (frame is not None and (feat in frame["features"])):
                                        prev[feat] = frame["features"][feat]
                                    if feat != "angle_vertical":
                                        row.append(prev[feat])
                                    if feat in ["re", "ratio_derivative", "gf"]:
                                        prev[feat] = 0
                                if (ip_set[i+1-DEFAULT_CONSEC_FRAMES] is not None and (feat in ip_set[i+1-DEFAULT_CONSEC_FRAMES]["features"])):
                                    prevprev[feat] = ip_set[i+1-DEFAULT_CONSEC_FRAMES]["features"][feat]
                            prevprev["re"] = 0
                            prevprev["ratio_derivative"] = 0
                            prevprev["gf"] = 0

                            row.append(int(df.iloc[i-zero-DEFAULT_CONSEC_FRAMES]["Tag"]))
                            row.append(prev["angle_vertical"])
                            if act in [1, 2, 3, 4, 5, 9]:
                                df_start = max(0, i-zero-2*DEFAULT_CONSEC_FRAMES+1)
                                df_quarter_end = min(i-zero-DEFAULT_CONSEC_FRAMES, df_start+DEFAULT_CONSEC_FRAMES//4)
                                if df.iloc[i-zero-DEFAULT_CONSEC_FRAMES]["Tag"] == act and act not in df.iloc[df_start: df_quarter_end]["Tag"].values:
                                    # for _ in range(3):
                                    final_data.append(row)
                                    if int(df.iloc[i-zero-DEFAULT_CONSEC_FRAMES]["Tag"]) == 11:
                                        print(sub, act, trial_num, i-zero-1)
                            else:
                                if i > zero+1+2*DEFAULT_CONSEC_FRAMES and sum(ip is not None for ip in ip_set[i+1-DEFAULT_CONSEC_FRAMES:i+1]) > 3*DEFAULT_CONSEC_FRAMES//4:
                                    if int(df.iloc[i-zero-DEFAULT_CONSEC_FRAMES]["Tag"]) == act:
                                        final_data.append(row)

                else:
                    f_q = alg1(trial, df)
                    try:
                        ip_set = f_q.get()[0]
                    except:
                        continue

                    zero = 1000 - len(trial) + 1 - (DEFAULT_CONSEC_FRAMES - 1)
                    zero = max(0, zero)
                    for i in range(zero, 1000 - len(trial) + 1):
                        ip_set[i] = ip_set[1000 - len(trial) + 1]

                    last = len(ip_set)
                    # if act == 6:
                    #     last -= (len(ip_set) - zero)//4
                    # elif act == 7:
                    #     last -= (len(ip_set) - zero)//3
                    # elif act == 8:
                    #     last -= 1*(len(ip_set) - zero)//3
                    # elif act == 10:
                    #     last -= (len(ip_set) - zero)//4

                    prevprev = {feat: 0 for feat in FEATURE_LIST}
                    for i in range(zero+1+DEFAULT_CONSEC_FRAMES, last):
                        row = []
                        prev = {feat: prevprev[feat] for feat in FEATURE_LIST}

                        if act == 11:
                            if ip_set[i] is not None:
                                for feat in FEATURE_LIST:
                                    if feat != "angle_vertical":
                                        row += [ip_set[i]["features"][feat] if feat in ip_set[i]["features"] else 0]*DEFAULT_CONSEC_FRAMES
                                row.append(act)
                                row.append(ip_set[i]["features"]["angle_vertical"])
                                final_data.append(row)
                            continue

                        for feat in FEATURE_LIST:
                            for frame in ip_set[i+1-DEFAULT_CONSEC_FRAMES:i+1]:
                                if (frame is not None and (feat in frame["features"])):
                                    prev[feat] = frame["features"][feat]
                                if feat != "angle_vertical":
                                    row.append(prev[feat])
                                if feat in ["re", "ratio_derivative", "gf"]:
                                    prev[feat] = 0
                            if (ip_set[i+1-DEFAULT_CONSEC_FRAMES] is not None and (feat in ip_set[i+1-DEFAULT_CONSEC_FRAMES]["features"])):
                                prevprev[feat] = ip_set[i+1-DEFAULT_CONSEC_FRAMES]["features"][feat]
                        prevprev["re"] = 0
                        prevprev["ratio_derivative"] = 0
                        prevprev["gf"] = 0

                        row.append(int(df.iloc[i-zero-DEFAULT_CONSEC_FRAMES]["Tag"]))
                        row.append(prev["angle_vertical"])
                        if act in [1, 2, 3, 4, 5, 9]:
                            df_start = max(0, i-zero-2*DEFAULT_CONSEC_FRAMES+1)
                            df_quarter_end = min(i-zero-DEFAULT_CONSEC_FRAMES, df_start+DEFAULT_CONSEC_FRAMES//4)
                            if df.iloc[i-zero-DEFAULT_CONSEC_FRAMES]["Tag"] == act and act not in df.iloc[df_start: df_quarter_end]["Tag"].values:
                                # for _ in range(10):
                                #     print(act)
                                final_data.append(row)
                                if int(df.iloc[i-zero-DEFAULT_CONSEC_FRAMES]["Tag"]) == 11:
                                    print(sub, act, trial_num, i-zero-1)
                        else:
                            if i > zero+1+2*DEFAULT_CONSEC_FRAMES and sum(ip is not None for ip in ip_set[i+1-DEFAULT_CONSEC_FRAMES:i+1]) > 3*DEFAULT_CONSEC_FRAMES//4:
                                if int(df.iloc[i-zero-DEFAULT_CONSEC_FRAMES]["Tag"]) == act:
                                    final_data.append(row)

        print(time.time()-t0)

    final_df = pd.DataFrame(final_data)
    # print(final_df[180].value_counts())
    final_df.to_csv(f'dataset/{csv_name}.csv', index=False, header=False)

    # for feat in FEATURE_LIST:
    #     final_df = pd.DataFrame(row[feat])
    #     final_df.to_csv(f'dataset/{feat}_data.csv', index=False, header=False)


if __name__ == '__main__':
    extract(1, 18, '2sec_multi_train_data_c1_allf_os')
    # extract(14, 16, '2sec_multi_cv_data_combined_allf_rbb')
    # extract(16, 18, '2sec_multi_test_data_c1_lr')
