from enum import IntEnum, unique
from typing import List
import cv2
import numpy as np


@unique
class CocoPart(IntEnum):
    """Body part locations in the 'coordinates' list."""
    Nose = 0
    LEye = 1
    REye = 2
    LEar = 3
    REar = 4
    LShoulder = 5
    RShoulder = 6
    LElbow = 7
    RElbow = 8
    LWrist = 9
    RWrist = 10
    LHip = 11
    RHip = 12
    LKnee = 13
    RKnee = 14
    LAnkle = 15
    RAnkle = 16


SKELETON_CONNECTIONS_COCO = [(0, 1, (210, 182, 247)), (0, 2, (127, 127, 127)), (1, 2, (194, 119, 227)),
                             (1, 3, (199, 199, 199)), (2, 4, (34, 189, 188)), (3, 5, (141, 219, 219)),
                             (4, 6, (207, 190, 23)), (5, 6, (150, 152, 255)), (5, 7, (189, 103, 148)),
                             (5, 11, (138, 223, 152)), (6, 8, (213, 176, 197)), (6, 12, (40, 39, 214)),
                             (7, 9, (75, 86, 140)), (8, 10, (148, 156, 196)), (11, 12, (44, 160, 44)),
                             (11, 13, (232, 199, 174)), (12, 14,
                                                         (120, 187, 255)), (13, 15, (180, 119, 31)),
                             (14, 16, (14, 127, 255))]


SKELETON_CONNECTIONS_5P = [('H', 'N', (210, 182, 247)), ('N', 'B', (210, 182, 247)), ('B', 'KL', (210, 182, 247)),
                           ('B', 'KR', (210, 182, 247)), ('KL', 'KR', (210, 182, 247))]

COLOR_ARRAY = [(210, 182, 247), (127, 127, 127), (194, 119, 227), (199, 199, 199), (34, 189, 188),
               (141, 219, 219), (207, 190, 23), (150, 152, 255), (189, 103, 148), (138, 223, 152)]

UNMATCHED_COLOR = (180, 119, 31)
# activity_dict = {
#     1.0: "Falling forward using hands",
#     2.0: "Falling forward using knees",
#     3: "Falling backwards",
#     4: "Falling sideward",
#     5: "Falling",
#     6: "Walking",
#     7: "Standing",
#     8: "Sitting",
#     9: "Picking up an object",
#     10: "Jumping",
#     11: "Laying",
#     12: "False Fall",
#     20: "None"
# }
activity_dict = {
    1.0: "Falling forward using hands",
    2.0: "Falling forward using knees",
    3: "Falling backwards",
    4: "Falling sideward",
    5: "FALL",
    6: "Normal",
    7: "Normal",
    8: "Normal",
    9: "Normal",
    10: "Normal",
    11: "Normal",
    12: "FALL Warning",
    20: "None"
}


def write_on_image(img: np.ndarray, text: str, color: List) -> np.ndarray:
    """Write text at the top of the image."""
    # Add a white border to top of image for writing text
    img = cv2.copyMakeBorder(src=img,
                             top=int(0.1 * img.shape[0]),
                             bottom=0,
                             left=0,
                             right=0,
                             borderType=cv2.BORDER_CONSTANT,
                             dst=None,
                             value=[255, 255, 255])
    for i, line in enumerate(text.split('\n')):
        y = 30 + i * 30
        cv2.putText(img=img,
                    text=line,
                    org=(0, y),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.7,
                    color=color,
                    thickness=2)

    return img


def visualise(img: np.ndarray, keypoint_sets: List, width: int, height: int, vis_keypoints: bool = False,
              vis_skeleton: bool = False, CocoPointsOn: bool = False) -> np.ndarray:
    """Draw keypoints/skeleton on the output video frame."""

    if CocoPointsOn:
        SKELETON_CONNECTIONS = SKELETON_CONNECTIONS_COCO
    else:
        SKELETON_CONNECTIONS = SKELETON_CONNECTIONS_5P

    if vis_keypoints or vis_skeleton:
        for keypoints in keypoint_sets:
            if not CocoPointsOn:
                keypoints = keypoints["keypoints"]

            if vis_skeleton:
                for p1i, p2i, color in SKELETON_CONNECTIONS:
                    if keypoints[p1i] is None or keypoints[p2i] is None:
                        continue

                    p1 = (int(keypoints[p1i][0] * width), int(keypoints[p1i][1] * height))
                    p2 = (int(keypoints[p2i][0] * width), int(keypoints[p2i][1] * height))

                    if p1 == (0, 0) or p2 == (0, 0):
                        continue

                    cv2.line(img=img, pt1=p1, pt2=p2, color=color, thickness=3)

    return img


def visualise_tracking(img: np.ndarray, keypoint_sets: List, width: int, height: int, num_matched: int, vis_keypoints: bool = False,
                       vis_skeleton: bool = False, CocoPointsOn: bool = False) -> np.ndarray:
    """Draw keypoints/skeleton on the output video frame."""

    if CocoPointsOn:
        SKELETON_CONNECTIONS = SKELETON_CONNECTIONS_COCO
    else:
        SKELETON_CONNECTIONS = SKELETON_CONNECTIONS_5P

    if vis_keypoints or vis_skeleton:
        for i, keypoints in enumerate(keypoint_sets):
            if keypoints is None:
                continue
            if not CocoPointsOn:
                keypoints = keypoints["keypoints"]
            if vis_skeleton:
                for p1i, p2i, color in SKELETON_CONNECTIONS:
                    if keypoints[p1i] is None or keypoints[p2i] is None:
                        continue

                    p1 = (int(keypoints[p1i][0] * width), int(keypoints[p1i][1] * height))
                    p2 = (int(keypoints[p2i][0] * width), int(keypoints[p2i][1] * height))

                    if p1 == (0, 0) or p2 == (0, 0):
                        continue
                    if i < num_matched:
                        color = COLOR_ARRAY[i % 10]
                    else:
                        color = UNMATCHED_COLOR

                    cv2.line(img=img, pt1=p1, pt2=p2, color=color, thickness=3)

    return img
