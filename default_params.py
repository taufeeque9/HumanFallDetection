DEFAULT_CONSEC_FRAMES = 36
MIN_THRESH = 0.5
TAKE_AVG_VALS_GF = False
HIST_THRESH = 0.2

HEAD_THRESHOLD = 1e-5
EMA_FRAMES = DEFAULT_CONSEC_FRAMES*3
EMA_BETA = 1/(EMA_FRAMES+1)
FEATURE_SCALAR = {"ratio_bbox": 1, "gf": 1, "angle_vertical": 1, "re": 1, "ratio_derivative": 1, "log_angle": 1}
FEATURE_LIST = ["ratio_bbox", "log_angle", "re", "ratio_derivative", "gf"]
# FEATURE_LIST = ["ratio_bbox", "log_angle", "re", "ratio_derivative", "gf", "angle_vertical"]
# FEATURE_LIST = ["re", "gf"]
# FEATURE_LIST = ["log_angle", "re", "gf"]
# FEATURE_LIST = ["log_angle", "re", "ratio_derivative", "gf"]
FRAME_FEATURES = 2
