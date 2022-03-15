import os
from itertools import product

import cv2
import numpy as np

SELECTED_FRAMES = [
    36,
    53,
    59,
    62,
    66,
    71,
    82
]


def overlay(background, foreground):
    # normalize alpha channels from 0-255 to 0-1
    alpha_background = background[:, :, 3] / 255.0
    alpha_foreground = foreground[:, :, 3] / 255.0

    # set adjusted colors
    for color in range(0, 3):
        background[:, :, color] = alpha_foreground * foreground[:, :, color] + \
            alpha_background * background[:, :, color] * (1 - alpha_foreground)

    # set adjusted alpha and denormalize back to 0-255
    background[:, :, 3] = (1 - (1 - alpha_foreground)
                           * (1 - alpha_background)) * 255

    return background


if __name__ == '__main__':
    vidcap = cv2.VideoCapture(
        '/home/figo/Develop/IC/sac_experiments/nao_videos/fcn_noisehead.mpg'
    )

    success, image = vidcap.read()

    i = 0
    while success:
        cv2.imwrite(f'/tmp/frame_{i}.png', image)
        os.system(
            f'convert /tmp/frame_{i}.png -fuzz 8% '
            f'-transparent white /tmp/frame_{i}.png'
        )

        i += 1

        success, image = vidcap.read()

    vidcap.release()

    if not SELECTED_FRAMES:
        exit()

    final_image = np.ones((232, 868, 4), np.uint8) * 255
    for i in SELECTED_FRAMES:
        current_image = cv2.imread(f'/tmp/frame_{i}.png', cv2.IMREAD_UNCHANGED)

        final_image = overlay(final_image, current_image)

    cv2.imwrite('final.png', final_image)
