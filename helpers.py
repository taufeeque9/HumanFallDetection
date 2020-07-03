import numpy as np
import matplotlib
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


def dist(ip1, ip2):
    ip1 = ip1[0]
    ip2 = ip2[0]
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


def get_hist(img, bbox, nbins=10):
    mask = Image.new('L', (img.shape[1], img.shape[0]), 0)
    ImageDraw.Draw(mask).polygon(bbox, outline=1, fill=1)
    mask = np.array(mask)
    hist = cv2.calcHist([img], [0, 1, 2], mask, [nbins, nbins, nbins], [0, 256, 0, 256, 0, 256])
    hist /= img.shape[0]*img.shape[1]

    return hist
