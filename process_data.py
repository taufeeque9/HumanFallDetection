if __name__ == "__main__":
    import fall_detector_restructured
    import sys
    import csv
    import os
    sub_start = 1
    sub_end = 2
    orig_sys_argv = sys.argv
    for act_id in range(1, 12):
        with open(f"dataset/Activity{act_id}/re.csv", "a", newline="") as reh, \
                open(f"dataset/Activity{act_id}/gf.csv", "a", newline="") as gfh:

            re_writer = csv.writer(reh)
            gf_writer = csv.writer(gfh)
            for sub_id in range(sub_start, sub_end):
                for trial_id in range(1, 4):
                    for cam_id in range(1, 3):
                        if os.path.exists(f'dataset/Activity{act_id}/Subject{sub_id}/Trial{trial_id}Cam{cam_id}.mp4'):
                            args = ['--plot_graph', '--checkpoint=resnet18',
                                    f'--video=dataset/Activity{act_id}/Subject{sub_id}/Trial{trial_id}Cam{cam_id}.mp4']
                            sys.argv = [orig_sys_argv[0]] + args
                            f = fall_detector_restructured.FallDetector()
                            re_matrix, gf_matrix = f.get_features()
                            # print(re_matrix)
                            re_writer.writerow(re_matrix[0])
                            gf_writer.writerow(gf_matrix[0])
