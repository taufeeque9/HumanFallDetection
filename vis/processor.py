import base64
import io
import time

import openpifpaf
import PIL
import torch
import numpy as np


class Processor(object):
    def __init__(self, width_height, args):
        self.width_height = width_height

        # Load model
        self.model_cpu, _ = openpifpaf.network.Factory().factory()
        self.model = self.model_cpu.to(args.device)
        self.processor = openpifpaf.decoder.factory(self.model_cpu.head_metas)
        # print(self.processor.device)
        self.device = args.device

    def get_bb(self, kp_set, score=None):
        bb_list = []
        for i in range(kp_set.shape[0]):
            x = kp_set[i, :15, 0]
            y = kp_set[i, :15, 1]
            v = kp_set[i, :15, 2]
            assert np.any(v > 0)
            if not np.any(v > 0):
                return None

            # keypoint bounding box
            x1, x2 = np.min(x[v > 0]), np.max(x[v > 0])
            y1, y2 = np.min(y[v > 0]), np.max(y[v > 0])
            if x2 - x1 < 5.0/self.width_height[0]:
                x1 -= 2.0/self.width_height[0]
                x2 += 2.0/self.width_height[0]
            if y2 - y1 < 5.0/self.width_height[1]:
                y1 -= 2.0/self.width_height[1]
                y2 += 2.0/self.width_height[1]

            bb_list.append(((x1, y1), (x2, y2)))

        # ax.add_patch(
        #     matplotlib.patches.Rectangle(
        #         (x1, y1), x2s - x1, y2 - y1, fill=False, color=color))
        #
        # if score:
        #     ax.text(x1, y1, '{:.4f}'.format(score), fontsize=8, color=color)
        return bb_list

    @staticmethod
    def keypoint_sets(annotations):
        keypoint_sets = [ann.data for ann in annotations]
        # scores = [ann.score() for ann in annotations]
        # assert len(scores) == len(keypoint_sets)
        if not keypoint_sets:
            return np.zeros((0, 17, 3))
        keypoint_sets = np.array(keypoint_sets)
        # scores = np.array(scores)

        return keypoint_sets

    def single_image(self, image):
        # image_bytes = io.BytesIO(base64.b64decode(b64image))
        # im = PIL.Image.open(image_bytes).convert('RGB')
        im = PIL.Image.fromarray(image)

        target_wh = self.width_height
        if (im.size[0] > im.size[1]) != (target_wh[0] > target_wh[1]):
            target_wh = (target_wh[1], target_wh[0])
        if im.size[0] != target_wh[0] or im.size[1] != target_wh[1]:
            # print(f'!!! have to resize image to {target_wh} from {im.size}')
            im = im.resize(target_wh, PIL.Image.BICUBIC)
        width_height = im.size

        start = time.time()
        preprocess = openpifpaf.transforms.Compose([
            openpifpaf.transforms.NormalizeAnnotations(),
            openpifpaf.transforms.CenterPadTight(16),
            openpifpaf.transforms.EVAL_TRANSFORM,
        ])
        # processed_image, _, __ = preprocess(im, [], None)
        processed_image = openpifpaf.datasets.PilImageList([im], preprocess=preprocess)[0][0]
        # processed_image = processed_image_cpu.contiguous().to(self.device, non_blocking=True)
        # print(f'preprocessing time {time.time() - start}')

        all_fields = self.processor.batch(self.model, torch.unsqueeze(
            processed_image.float(), 0), device=self.device)[0]
        keypoint_sets = self.keypoint_sets(all_fields)

        # Normalize scale
        keypoint_sets[:, :, 0] /= processed_image.shape[2]
        keypoint_sets[:, :, 1] /= processed_image.shape[1]

        bboxes = self.get_bb(keypoint_sets)
        return keypoint_sets, bboxes, width_height
