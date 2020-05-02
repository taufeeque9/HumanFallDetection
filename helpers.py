import numpy as np


def pop_and_add(l, val, max_length):
    if len(l) == max_length:
        l.pop(0)
    l.append(val)


def last_ip(ips):
    for ip in reversed(ips):
        if ip is not None:
            return ip


def dist(ip1, ip2):
    return np.sqrt(np.sum((ip1['H']-ip2['H'])**2 + (ip1['N']-ip2['N'])**2 + (ip1['B']-ip2['B'])**2))
