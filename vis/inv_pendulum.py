from .visual import CocoPart
import numpy as np
from helpers import *
from default_params import *


def match_ip(ip_set, new_ips, lstm_set, num_matched, consecutive_frames=DEFAULT_CONSEC_FRAMES):
    len_ip_set = len(ip_set)
    added = [False for _ in range(len_ip_set)]

    new_len_ip_set = len_ip_set
    for new_ip in new_ips:
        if not is_valid(new_ip):
            continue
        # assert valid_candidate_hist(new_ip)
        cmin = [MIN_THRESH, -1]
        for i in range(len_ip_set):
            if not added[i] and dist(last_ip(ip_set[i])[0], new_ip) < cmin[0]:
                # here add dome condition that last_ip(ip_set[0] >-5 or someting)
                cmin[0] = dist(last_ip(ip_set[i])[0], new_ip)
                cmin[1] = i

        if cmin[1] == -1:
            ip_set.append([None for _ in range(consecutive_frames - 1)] + [new_ip])
            lstm_set.append([None, 0, 0, 0])  # Initial hidden state of lstm is None
            new_len_ip_set += 1

        else:
            added[cmin[1]] = True
            pop_and_add(ip_set[cmin[1]], new_ip, consecutive_frames)

    new_matched = num_matched

    removed_indx = []
    removed_match = []

    for i in range(len(added)):
        if not added[i]:
            pop_and_add(ip_set[i], None, consecutive_frames)
        if ip_set[i] == [None for _ in range(consecutive_frames)]:
            if i < num_matched:
                new_matched -= 1
                removed_match.append(i)

            new_len_ip_set -= 1
            removed_indx.append(i)

    for i in sorted(removed_indx, reverse=True):
        ip_set.pop(i)
        lstm_set.pop()

    return new_matched, new_len_ip_set, removed_match


def extend_vector(p1, p2, l):
    p1 += (p1-p2)*l/(2*np.linalg.norm((p1-p2), 2))
    p2 -= (p1-p2)*l/(2*np.linalg.norm((p1-p2), 2))
    return p1, p2


def perp(a):
    b = np.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b

# line segment a given by endpoints a1, a2
# line segment b given by endpoints b1, b2
# return


def seg_intersect(a1, a2, b1, b2):
    da = a2-a1
    db = b2-b1
    dp = a1-b1
    dap = perp(da)
    denom = np.dot(dap, db)
    num = np.dot(dap, dp)
    return (num / denom.astype(float))*db + b1


def get_kp(kp):
    threshold1 = 5e-3

    # dict of np arrays of coordinates
    inv_pend = {}
    # print(type(kp[CocoPart.LEar]))
    numx = (kp[CocoPart.LEar][2]*kp[CocoPart.LEar][0] + kp[CocoPart.LEye][2]*kp[CocoPart.LEye][0] +
            kp[CocoPart.REye][2]*kp[CocoPart.REye][0] + kp[CocoPart.REar][2]*kp[CocoPart.REar][0])
    numy = (kp[CocoPart.LEar][2]*kp[CocoPart.LEar][1] + kp[CocoPart.LEye][2]*kp[CocoPart.LEye][1] +
            kp[CocoPart.REye][2]*kp[CocoPart.REye][1] + kp[CocoPart.REar][2]*kp[CocoPart.REar][1])
    den = kp[CocoPart.LEar][2] + kp[CocoPart.LEye][2] + kp[CocoPart.REye][2] + kp[CocoPart.REar][2]

    if den < HEAD_THRESHOLD:
        inv_pend['H'] = None
    else:
        inv_pend['H'] = np.array([numx/den, numy/den])

    if all([kp[CocoPart.LShoulder], kp[CocoPart.RShoulder],
            kp[CocoPart.LShoulder][2] > threshold1, kp[CocoPart.RShoulder][2] > threshold1]):
        inv_pend['N'] = np.array([(kp[CocoPart.LShoulder][0]+kp[CocoPart.RShoulder][0])/2,
                                  (kp[CocoPart.LShoulder][1]+kp[CocoPart.RShoulder][1])/2])
    else:
        inv_pend['N'] = None

    if all([kp[CocoPart.LHip], kp[CocoPart.RHip],
            kp[CocoPart.LHip][2] > threshold1, kp[CocoPart.RHip][2] > threshold1]):
        inv_pend['B'] = np.array([(kp[CocoPart.LHip][0]+kp[CocoPart.RHip][0])/2,
                                  (kp[CocoPart.LHip][1]+kp[CocoPart.RHip][1])/2])
    else:
        inv_pend['B'] = None

    if kp[CocoPart.LKnee] is not None and kp[CocoPart.LKnee][2] > threshold1:
        inv_pend['KL'] = np.array([kp[CocoPart.LKnee][0], kp[CocoPart.LKnee][1]])
    else:
        inv_pend['KL'] = None

    if kp[CocoPart.RKnee] is not None and kp[CocoPart.RKnee][2] > threshold1:
        inv_pend['KR'] = np.array([kp[CocoPart.RKnee][0], kp[CocoPart.RKnee][1]])
    else:
        inv_pend['KR'] = None

    if inv_pend['B'] is not None:
        if inv_pend['N'] is not None:
            height = np.linalg.norm(inv_pend['N'] - inv_pend['B'], 2)
            LS, RS = extend_vector(np.asarray(kp[CocoPart.LShoulder][:2]),
                                   np.asarray(kp[CocoPart.RShoulder][:2]), height/4)
            LB, RB = extend_vector(np.asarray(kp[CocoPart.LHip][:2]),
                                   np.asarray(kp[CocoPart.RHip][:2]), height/3)
            ubbox = (LS, RS, RB, LB)

            if inv_pend['KL'] is not None and inv_pend['KR'] is not None:
                lbbox = (LB, RB, inv_pend['KR'], inv_pend['KL'])
            else:
                lbbox = ([0, 0], [0, 0])
                #lbbox = None
        else:
            ubbox = ([0, 0], [0, 0])
            #ubbox = None
            if inv_pend['KL'] is not None and inv_pend['KR'] is not None:
                lbbox = (np.array(kp[CocoPart.LHip][:2]), np.array(kp[CocoPart.RHip][:2]),
                         inv_pend['KR'], inv_pend['KL'])
            else:
                lbbox = ([0, 0], [0, 0])
                #lbbox = None
    else:
        ubbox = ([0, 0], [0, 0])
        lbbox = ([0, 0], [0, 0])
        #ubbox = None
        #lbbox = None
    # condition = (inv_pend["H"] is None) and (inv_pend['N'] is not None and inv_pend['B'] is not None)
    # if condition:
    #     print("half disp")

    return inv_pend, ubbox, lbbox


def get_angle(v0, v1):
    return np.math.atan2(np.linalg.det([v0, v1]), np.dot(v0, v1))


def is_valid(ip):

    assert ip is not None

    ip = ip["keypoints"]
    return (ip['B'] is not None and ip['N'] is not None and ip['H'] is not None)


def get_rot_energy(ip0, ip1):
    t = ip1["time"] - ip0["time"]
    ip0 = ip0["keypoints"]
    ip1 = ip1["keypoints"]
    m1 = 1
    m2 = 5
    m3 = 5
    energy = 0
    den = 0
    N1 = ip1['N'] - ip1['B']
    N0 = ip0['N'] - ip0['B']
    d2sq = N1.dot(N1)
    w2sq = (get_angle(N0, N1)/t)**2
    energy += m2*d2sq*w2sq

    den += m2*d2sq
    H1 = ip1['H'] - ip1['B']
    H0 = ip0['H'] - ip0['B']
    d1sq = H1.dot(H1)
    w1sq = (get_angle(H0, H1)/t)**2
    energy += m1*d1sq*w1sq
    den += m1*d1sq

    energy = energy/(2*den)
    # energy = energy/2
    return energy


def get_angle_vertical(v):
    return np.math.atan2(-v[0], -v[1])


def get_gf(ip0, ip1, ip2):
    t1 = ip1["time"] - ip0["time"]
    t2 = ip2["time"] - ip1["time"]
    ip0 = ip0["keypoints"]
    ip1 = ip1["keypoints"]
    ip2 = ip2["keypoints"]

    m1 = 1
    m2 = 15
    g = 10
    H2 = ip2['H'] - ip2['N']
    H1 = ip1['H'] - ip1['N']
    H0 = ip0['H'] - ip0['N']
    d1 = np.sqrt(H1.dot(H1))
    theta_1_plus_2_2 = get_angle_vertical(H2)
    theta_1_plus_2_1 = get_angle_vertical(H1)
    theta_1_plus_2_0 = get_angle_vertical(H0)
    # print("H: ",H0,H1,H2)
    N2 = ip2['N'] - ip2['B']
    N1 = ip1['N'] - ip1['B']
    N0 = ip0['N'] - ip0['B']
    d2 = np.sqrt(N1.dot(N1))
    # print("N: ",N0,N1,N2)
    theta_2_2 = get_angle_vertical(N2)
    theta_2_1 = get_angle_vertical(N1)
    theta_2_0 = get_angle_vertical(N0)
    #print("theta_2_2:",theta_2_2,"theta_2_1:",theta_2_1,"theta_2_0:",theta_2_0,sep=", ")
    theta_1_0 = theta_1_plus_2_0 - theta_2_0
    theta_1_1 = theta_1_plus_2_1 - theta_2_1
    theta_1_2 = theta_1_plus_2_2 - theta_2_2

    # print("theta1: ",theta_1_0,theta_1_1,theta_1_2)
    # print("theta2: ",theta_2_0,theta_2_1,theta_2_2)

    theta2 = theta_2_1
    theta1 = theta_1_1

    del_theta1_0 = (get_angle(H0, H1))/t1
    del_theta1_1 = (get_angle(H1, H2))/t2

    del_theta2_0 = (get_angle(N0, N1))/t1
    del_theta2_1 = (get_angle(N1, N2))/t2
    # print("del_theta2_1:",del_theta2_1,"del_theta2_0:",del_theta2_0,sep=",")
    del_theta1 = 0.5 * (del_theta1_1 + del_theta1_0)
    del_theta2 = 0.5 * (del_theta2_1 + del_theta2_0)

    doubledel_theta1 = (del_theta1_1 - del_theta1_0) / 0.5*(t1 + t2)
    doubledel_theta2 = (del_theta2_1 - del_theta2_0) / 0.5*(t1 + t2)
    # print("doubledel_theta2:",doubledel_theta2)

    d1 = d1/d2
    d2 = 1
    # print("del_theta",del_theta1,del_theta2)
    # print("doubledel_theta",doubledel_theta1,doubledel_theta2)

    Q_RD1 = 0
    Q_RD1 += m1 * d1 * doubledel_theta1 * doubledel_theta1
    Q_RD1 += (m1*d1*d1 + m1*d1*d2*np.cos(theta1))*doubledel_theta2
    Q_RD1 += m1*d1*d2*np.sin(theta1)*del_theta2*del_theta2
    Q_RD1 -= m1*g*d2*np.sin(theta1+theta2)

    Q_RD2 = 0
    Q_RD2 += (m1*d1*d1 + m1*d1*d2*np.cos(theta1))*doubledel_theta1
    Q_RD2 += ((m1+m2)*d2*d2 + m1*d1*d1 + 2*m1*d1*d2*np.cos(theta1))*doubledel_theta2
    Q_RD2 -= 2*m1*d1*d2*np.sin(theta1)*del_theta2*del_theta1 + m1*d1 * \
        d2*np.sin(theta1)*del_theta1*del_theta1
    Q_RD2 -= (m1 + m2)*g*d2*np.sin(theta2) + m1*g*d1*np.sin(theta1 + theta2)

    # print("Energy: ", Q_RD1 + Q_RD2)
    return Q_RD1 + Q_RD2


def get_height_bbox(ip):
    bbox = ip["box"]
    assert(type(bbox == np.ndarray))
    diff_box = bbox[1] - bbox[0]
    return diff_box[1]


def get_ratio_bbox(ip):
    bbox = ip["box"]
    assert(type(bbox == np.ndarray))
    diff_box = bbox[1] - bbox[0]
    if diff_box[1] == 0:
        diff_box[1] += 1e5*diff_box[0]
    assert(np.any(diff_box > 0))
    ratio = diff_box[0]/diff_box[1]
    return ratio


def get_ratio_derivative(ip0, ip1):
    ratio_der = None
    time = ip1["time"] - ip0["time"]
    diff_box = ip1["features"]["ratio_bbox"] - ip0["features"]["ratio_bbox"]
    assert time != 0
    ratio_der = diff_box/time

    return ratio_der


def match_ip2(matched_ip_set, unmatched_ip_set, new_ips, re_matrix, gf_matrix, consecutive_frames=DEFAULT_CONSEC_FRAMES):
    len_matched_ip_set = len(matched_ip_set)
    added_matched = [False for _ in range(len_matched_ip_set)]
    len_unmatched_ip_set = len(unmatched_ip_set)
    added_unmatched = [False for _ in range(len_unmatched_ip_set)]
    for new_ip in new_ips:
        if not is_valid(new_ip):
            continue
        cmin = [MIN_THRESH, -1]
        connected_set = None
        connected_added = None
        for i in range(len_matched_ip_set):
            if not added_matched[i] and dist(last_ip(matched_ip_set[i])[0], new_ip) < cmin[0]:
                # here add dome condition that last_ip(ip_set[0] >-5 or someting)
                cmin[0] = dist(last_ip(matched_ip_set[i])[0], new_ip)
                cmin[1] = i
                connected_set = matched_ip_set
                connected_added = added_matched
        for i in range(len_unmatched_ip_set):
            if not added_unmatched[i] and dist(last_ip(unmatched_ip_set[i])[0], new_ip) < cmin[0]:
                # here add dome condition that last_ip(ip_set[0] >-5 or someting)
                cmin[0] = dist(last_ip(unmatched_ip_set[i])[0], new_ip)
                cmin[1] = i
                connected_set = unmatched_ip_set
                connected_added = added_unmatched

        if cmin[1] == -1:
            unmatched_ip_set.append([None for _ in range(consecutive_frames - 1)] + [new_ip])
            # re_matrix.append([])
            # gf_matrix.append([])

        else:
            connected_added[cmin[1]] = True
            pop_and_add(connected_set[cmin[1]], new_ip, consecutive_frames)

    i = 0
    while i < len(added_matched):
        if not added_matched[i]:
            pop_and_add(matched_ip_set[i], None, consecutive_frames)
            if matched_ip_set[i] == [None for _ in range(consecutive_frames)]:
                matched_ip_set.pop(i)
                # re_matrix.pop(i)
                # gf_matrix.pop(i)
                added_matched.pop(i)
                continue
        i += 1
