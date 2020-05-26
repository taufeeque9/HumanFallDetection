from visual import CocoPart
import numpy as np
from helpers import *
from default_params import *

def match_ip(ip_set, new_ips, re_matrix, gf_matrix, consecutive_frames=DEFAULT_CONSEC_FRAMES):
    len_ip_set = len(ip_set)
    added = [False for _ in range(len_ip_set)]
    for new_ip in new_ips:
        if not is_valid(new_ip):
            continue
        cmin = [MIN_THRESH, -1]
        for i in range(len_ip_set):
            if not added[i] and dist(last_ip(ip_set[i])[0], new_ip) < cmin[0]:
                # here add dome condition that last_ip(ip_set[0] >-5 or someting)
                cmin[0] = dist(last_ip(ip_set[i])[0], new_ip)
                cmin[1] = i

        if cmin[1] == -1:
            ip_set.append([None for _ in range(consecutive_frames - 1)] + [new_ip])
            re_matrix.append([0 for _ in range(consecutive_frames )])
            gf_matrix.append([0 for _ in range(consecutive_frames )])

        else:
            added[cmin[1]] = True
            pop_and_add(ip_set[cmin[1]], new_ip, consecutive_frames)

    i = 0
    while i < len(added):
        if not added[i]:
            pop_and_add(ip_set[i], None, consecutive_frames)
            if ip_set[i] == [None for _ in range(consecutive_frames)]:
                ip_set.pop(i)
                re_matrix.pop(i)
                gf_matrix.pop(i)
                added.pop(i)
                continue
        i += 1

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

    if den - kp[CocoPart.LEar][2] < 30*threshold1 or den - kp[CocoPart.LEye][2] < 30*threshold1 or \
       den - kp[CocoPart.REar][2] < 30*threshold1 or den - kp[CocoPart.REye][2] < 30*threshold1:
        inv_pend['H'] = None
    else:
        inv_pend['H'] = np.array([numx/den, numy/den])

    if all([kp[CocoPart.LShoulder] is not None, kp[CocoPart.RShoulder] is not None,
            kp[CocoPart.LShoulder][2] > threshold1, kp[CocoPart.RShoulder][2] > threshold1]):
        inv_pend['N'] = np.array([(kp[CocoPart.LShoulder][0]+kp[CocoPart.RShoulder][0])/2,
                                  (kp[CocoPart.LShoulder][1]+kp[CocoPart.RShoulder][1])/2])
    else:
        inv_pend['N'] = None

    if all([kp[CocoPart.LHip] is not None, kp[CocoPart.RHip] is not None,
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

    return inv_pend


def get_angle(v0, v1):
    return np.math.atan2(np.linalg.det([v0, v1]), np.dot(v0, v1))


def is_valid(ip):
    ip = ip[0]
    return (ip['B'] is not None and ip['N'] is not None and ip['H'] is not None)


def get_rot_energy(ip0, ip1):
    t = ip1[1] - ip0[1]
    ip0 = ip0[0]
    ip1 = ip1[0]
    m1 = 1
    m2 = 5
    m3 = 5
    energy = 0

    N1 = ip1['N'] - ip1['B']
    N0 = ip0['N'] - ip0['B']
    d2sq = N1.dot(N1)
    w2sq = (get_angle(N0, N1)/t)**2
    energy += m2*d2sq*w2sq

    H1 = ip1['H'] - ip1['B']
    H0 = ip0['H'] - ip0['B']
    d1sq = H1.dot(H1)
    w1sq = (get_angle(H0, H1)/t)**2
    energy += m1*d1sq*w1sq

    if ip0['KL'] is not None and ip0['KR']is not None:
        if ip1['KL'] is not None and ip1['KR']is not None:
            K1 = (ip1['KL'] + ip1['KR'])/2 - ip1['B']
            K0 = (ip0['KL'] + ip0['KR'])/2 - ip0['B']
            d3sq = K1.dot(K1)
            w3sq = (get_angle(K0, K1)/t)**2
            energy += m3*d3sq*w3sq

    energy = energy/(2*d2sq)
    # energy = energy/2
    return energy


def get_angle_vertical(v):
    return np.math.atan2(v[0], v[1])


def get_gf(ip0, ip1, ip2, t1=1, t2=1):
    t1 = ip1[1] - ip0[1]
    t2 = ip2[1] - ip1[1]
    ip0 = ip0[0]
    ip1 = ip1[0]
    ip2 = ip2[0]

    m1 = 1
    m2 = 5
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

    del_theta1 = 0.5 * (del_theta1_1 + del_theta1_0)
    del_theta2 = 0.5 * (del_theta2_1 + del_theta2_0)

    doubledel_theta1 = (del_theta1_1 - del_theta1_0) / 0.5*(t1 + t2)
    doubledel_theta2 = (del_theta2_1 - del_theta2_0) / 0.5*(t1 + t2)

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
