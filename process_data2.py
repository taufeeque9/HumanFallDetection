if __name__ == "__main__":
    import fall_detector
    import sys
    import csv
    import os
    import joblib
    import time
    sub_start = 15
    sub_end = 18
    orig_sys_argv = sys.argv
    for act_id in range(1, 12):
        for sub_id in range(sub_start, sub_end):
            dl = []
            t0 = time.time()
            for trial_id in range(1, 4):
                for cam_id in range(1, 3):
                    if os.path.exists(f'dataset/Activity{act_id}/Subject{sub_id}/Trial{trial_id}Cam1.mp4'):
                        args = ['--coco_points', f'--video=dataset/Activity{act_id}/Subject{sub_id}/Trial{trial_id}Cam{cam_id}.mp4']
                        sys.argv = [orig_sys_argv[0]] + args
                        f = fall_detector.FallDetector()
                        q1 = f.begin_mixed()
                        dl.append(q1)

            joblib.dump(dl, f'dataset/Activity{act_id}/Subject{sub_id}/coco.kps', True)
            print(time.time()-t0)
