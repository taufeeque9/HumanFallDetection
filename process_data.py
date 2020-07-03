if __name__ == "__main__":
    import fall_detector
    import sys
    import csv
    import os
    import itertools
    from matplotlib import pyplot as plt
    import matplotlib.image as mpimg

    def get_id(n):
        sub = n//6 + 1
        trial = (n % 6)//2 + 1
        cam = (n % 6) % 2 + 1
        return sub, trial, cam

    def plot_data(i, start=0, end=24):
        maxx = 0
        fig, ax = plt.subplots(
            nrows=2, ncols=3, figsize=(20, 10))

        plt.tick_params(
            axis='y',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False)  # labels along the bottom edge are off

        with open(f"dataset/Activity{i}/re.csv", "r") as reh, \
                open(f"dataset/Activity{i}/gf.csv", "r") as gfh:
            re_reader = csv.reader(reh)
            gf_reader = csv.reader(gfh)
            rerows = itertools.islice(re_reader, start, end)
            gfrows = itertools.islice(gf_reader, start, end)
            c1 = mpimg.imread("dataset/cam1.png")
            c2 = mpimg.imread("dataset/cam2.png")
            ax[0, 2].imshow(c1)
            ax[1, 2].imshow(c2)

            for ind, row in enumerate(rerows):
                maxx = len(row) if len(row) > maxx else maxx
                row = [float(v) for v in reversed(row)]
                sub, tri, cam = get_id(start + ind)
                lab = f"Sub{sub}Trial{tri}"
                if ind % 2 == 0:
                    ax[0, 0].plot(row, label=lab)
                else:
                    ax[1, 0].plot(row, label=lab)

            for ind, row in enumerate(gfrows):
                row = [float(v) for v in reversed(row)]
                sub, tri, cam = get_id(start + ind)
                lab = f"Sub{sub}Trial{tri}"
                row = [float(v) for v in row]
                if ind % 2 == 0:
                    ax[0, 1].plot(row, label=lab)
                else:
                    ax[1, 1].plot(row, label=lab)

        ax[0, 0].set_title('Rotation Force')
        ax[0, 1].set_title('General Energy')
        ax[0, 2].set_title('Camera Postion')
        ax[0, 0].set_ylabel('Cam 1', rotation=0, size='large', labelpad=25)
        ax[1, 0].set_ylabel('Cam 2', rotation=0, size='large', labelpad=10)
        print(maxx)
        ax[0, 2].set_yticks([])
        ax[1, 2].set_yticks([])
        ax[0, 2].set_xticks([])
        ax[1, 2].set_xticks([])
        [axis.legend() for axis in ax.ravel()]
        [axis.set_xlim(maxx, 0) for raxis in ax for axis in raxis[:2]]
        fig.tight_layout()
        from pathlib import Path
        plot_file = Path(f"dataset/Activity{i}/Subject{get_id(start)[0]}/feature_plot.png")
        if plot_file.is_file():
            os.remove(f"dataset/Activity{i}/Subject{get_id(start)[0]}/feature_plot.png")
        fig.savefig(f"dataset/Activity{i}/Subject{get_id(start)[0]}/feature_plot.png")
        plt.show()

    def generate_data():
        sub_start = 2
        sub_end = 4
        orig_sys_argv = sys.argv
        for act_id in range(1, 11):
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
                                f = fall_detector.FallDetector()
                                re_matrix, gf_matrix = f.get_features()
                                # print(re_matrix)
                                re_writer.writerow(re_matrix)
                                gf_writer.writerow(gf_matrix)

    # if len(sys.argv) > 1 and '--activity' in sys.argv[1]:
    #     if len(sys.argv) > 2 and '--sub_range' in sys.argv[2]:
    #         range = [int(f) for f in sys.argv[2].split('=')[1].split(',')]
    #         plot_data(int(sys.argv[1].split('activity')[1]), (range[0]-1)*6, (range[1]-1)*6)

    for act_id in range(1, 2):
        for sub_id in range(1, 2):
            plot_data(act_id, (sub_id-1)*6, (sub_id-1)*6+6)
