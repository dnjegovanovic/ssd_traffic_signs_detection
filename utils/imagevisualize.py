import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class ImageVisualizer(object):
    """
    Class for visualizing image
    """

    def __init__(self, idx_to_name, class_colors=None, save_dir=None):
        self.idx_to_name = idx_to_name
        if class_colors is None or len(class_colors) != len(self.idx_to_name):
            self.class_colors = [[0, 255, 0]] * len(self.idx_to_name)
        else:
            self.class_colors = class_colors

        if save_dir is None:
            self.save_dir = './'
        else:
            self.save_dir = save_dir

        os.makedirs(self.save_dir, exist_ok=True)

    def save_image(self, img, boxes, labels, name):
        """
        Method to draw boxes and labels
        then save to dir
        """
        plt.figure()
        fig, ax = plt.subplots(1)
        ax.imshow(img)
        save_path = os.path.join(self.save_dir, name)
        for i, box in enumerate(boxes):
            idx = labels[i] #- 1
            cls_name = self.idx_to_name[idx]
            if len(cls_name) == 0:
                cls_name = 'temp_empy'
            top_left = (box[0], box[1])
            bot_right = (box[2], box[3])
            ax.add_patch(patches.Rectangle(
                (box[0], box[1]),
                box[2] - box[0], box[3] - box[1],
                linewidth=2, edgecolor=(0., 1., 0.),
                facecolor="none"))
            plt.text(
                box[0],
                box[1],
                s=cls_name,
                color="white",
                verticalalignment="top",
                bbox={"color": (0., 1., 0.), "pad": 0},
                fontsize = 6
            )

        plt.axis("off")
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.0)
        plt.close('all')






