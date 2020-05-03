from visual import CocoPart
import numpy as np


def get_kp(kp):
    threshold1 = 1e-4
    # dict of np arrays of coordinates
    inv_pend = {}
    # print(type(kp[CocoPart.LEar]))
    numx = (kp[CocoPart.LEar][2]*kp[CocoPart.LEar][0] + kp[CocoPart.LEye][2]*kp[CocoPart.LEye][0] +
            kp[CocoPart.REye][2]*kp[CocoPart.REye][0] + kp[CocoPart.REar][2]*kp[CocoPart.REar][0])
    numy = (kp[CocoPart.LEar][2]*kp[CocoPart.LEar][1] + kp[CocoPart.LEye][2]*kp[CocoPart.LEye][1] +
            kp[CocoPart.REye][2]*kp[CocoPart.REye][1] + kp[CocoPart.REar][2]*kp[CocoPart.REar][1])
    den = kp[CocoPart.LEar][2] + kp[CocoPart.LEye][2] + kp[CocoPart.REye][2] + kp[CocoPart.REar][2]

    if den > 4*threshold1:
        inv_pend['H'] = np.array([numx/den, numy/den])
    else:
        inv_pend['H'] = None

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
    return (ip['B'] is not None and ip['N'] is not None and ip['H'] is not None)


def get_rot_energy(ip0, ip1, t=1):
    energy = 0
    H1 = ip1['H'] - ip1['B']
    H0 = ip0['H'] - ip0['B']
    rsq = H1.dot(H1)
    wsq = (get_angle(H0, H1)/t)**2
    energy += rsq*wsq
    N1 = ip1['N'] - ip1['B']
    N0 = ip0['N'] - ip0['B']
    rsq = N1.dot(N1)
    wsq = (get_angle(N0, N1)/t)**2
    energy += rsq*wsq

    return energy/2

def get_angle_vertical(v):
    return np.math.atan2(v[0],v[1])

def get_gf(ip0,ip1,ip2,t1=1,t2=1):
    
    m1 = 1
    m2 = 1
    g = 10
    H2 = ip2['H'] - ip2['N']
    H1 = ip1['H'] - ip1['N']
    H0 = ip0['H'] - ip0['N']
    d1 = H1.dot(H1)
    theta_1_plus_2_2 = get_angle_vertical(H2)
    theta_1_plus_2_1 = get_angle_vertical(H1) 
    theta_1_plus_2_0 = get_angle_vertical(H0)
    N2 = ip2['N'] - ip2['B']
    N1 = ip1['N'] - ip1['B']
    N0 = ip0['N'] - ip0['B']
    d2 = N1.dot(N1)
    theta_2_2 = get_angle_vertical(N2)
    theta_2_1 = get_angle_vertical(N1)
    theta_2_0 = get_angle_vertical(N0)

    theta_1_0 = theta_1_plus_2_0 - theta_2_0
    theta_1_1 = theta_1_plus_2_1 - theta_2_1
    theta_1_2 = theta_1_plus_2_2 - theta_2_2


    theta2 = theta_2_1
    theta1 = theta_1_1

    del_theta1_0 = (theta_1_1 - theta_1_0)/t1
    del_theta1_1 = (theta_1_2 - theta_1_1)/t2

    del_theta2_0 = (theta_2_1 - theta_2_0)/t1
    del_theta2_1 = (theta_2_2 - theta_2_1)/t2

    del_theta1 = 0.5 * ( del_theta1_1 + del_theta1_0 )
    del_theta2 = 0.5 * ( del_theta2_1 + del_theta2_0 )

    doubledel_theta1 = (del_theta1_1 - del_theta1_0) / 0.5*(t1 + t2)
    doubledel_theta2 = (del_theta2_1 - del_theta2_0) / 0.5*(t1 + t2)

    Q_RD1 = 0
    Q_RD1 += m1 * d1* doubledel_theta1 * doubledel_theta1
    Q_RD1 += (m1*d1*d1 + m1*d1*d2*np.cos(theta1))*doubledel_theta2
    Q_RD1 += m1*d1*d2*np.sin(theta1)*del_theta2*del_theta2
    Q_RD1 -= m1*g*d2*np.sin(theta1+theta2)

    Q_RD2 = 0
    Q_RD2 += (m1*d1*d1 + m1*d1*d2*np.cos(theta1))*doubledel_theta1
    Q_RD2 += ((m1+m2)*d2*d2 + m1*d1*d1 + 2*m1*d1*d2*np.cos(theta1))*doubledel_theta2
    Q_RD2 -= 2*m1*d1*d2*np.sin(theta1)*del_theta2*del_theta1 + m1*d1*d2*np.sin(theta1)*del_theta1*del_theta1
    Q_RD2 -= (m1 + m2)*g*d2*np.sin(theta2) + m1*g*d1*np.sin(theta1 + theta2)

    return Q_RD1 + Q_RD2


    