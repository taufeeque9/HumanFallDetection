import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import cv2
from PIL import Image, ImageDraw


def pop_and_add(l, val, max_length):
    if len(l) == max_length:
        l.pop(0)
    l.append(val)


def last_ip(ips):
    for i, ip in enumerate(reversed(ips)):
        if ip is not None:
            return ip, len(ips) - i

def last_valid_hist(ips):
    for i,ip in enumerate(reversed(ips)):
        if valid_candidate_hist(ip):
            return ip

def dist(ip1, ip2):
    ip1 = ip1["keypoints"]
    ip2 = ip2["keypoints"]
    return np.sqrt(np.sum((ip1['N']-ip2['N'])**2 + (ip1['B']-ip2['B'])**2))


def move_figure(f, x, y):
    """Move figure's upper left corner to pixel (x, y)"""
    backend = matplotlib.get_backend()
    if backend == 'TkAgg':
        f.canvas.manager.window.wm_geometry("+%d+%d" % (x, y))
    elif backend == 'WXAgg':
        f.canvas.manager.window.SetPosition((x, y))
    else:
        # This works for QT and GTK
        # You can also use window.setGeometry
        f.canvas.manager.window.move(x, y)

def valid_candidate_hist(ip):
    if ip is not None:
        return ip["up_hist"] is not None
    else:
        return False

def dist_hist(ips1,ips2):

    ip1 = last_valid_hist(ips1)
    ip2 = last_valid_hist(ips2)

    uhist1 = ip1["up_hist"]
    uhist2 = ip2["up_hist"]

    assert uhist1 is not None
    assert uhist2 is not None

    assert type(uhist1) == np.ndarray

    return np.sum(np.absolute(uhist1-uhist2))

def get_hist(img, bbox, nbins=3):

    if not np.any(bbox):
        return None

    mask = Image.new('L', (img.shape[1], img.shape[0]), 0)
    ImageDraw.Draw(mask).polygon(list(bbox.flatten()), outline=1, fill=1)
    mask = np.array(mask)
    hist = cv2.calcHist([img], [0, 1], mask, [nbins, 2*nbins], [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=1, norm_type=cv2.NORM_L1)


    return hist
