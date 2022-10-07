import os
import math
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from skimage import morphology
from .common_utils import (
    get_image,
    crop_image,
    pil_to_np,
    np_to_torch,
)


def show_img(img, size=5):
    plt.figure(figsize=(size, size))
    plt.imshow(img, cmap="gray")
    plt.axis("off")
    plt.show()


def get_max_distance(image):
    if image.ndim == 2:
        h, w = image.shape
    else:
        h, w, _ = image.shape

    return math.sqrt(h**2 + w**2)


def get_thin_points(image):
    if image.dtype != "uint8":
        image = cv2.normalize(
            image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U
        )
    if len(image.shape) != 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(
        image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    binary_image = binary_image.astype("float32")
    thined = morphology.thin(binary_image)
    x, y = np.where(thined == True)
    points = np.array((x, y)).T
    return points


def similarity(I, I_gt, threshold=0.5):
    points_gt = get_thin_points(I_gt)
    points = get_thin_points(I)
    max_d = get_max_distance(I)
    cut_point = threshold * 0.01 * max_d
    # ------------- Reconstruction -----
    kdtree = KDTree(points)
    d, _ = kdtree.query(points_gt)
    recons = round(
        (len(np.where(d < cut_point)[0]) / len(points_gt)) * 100
    )
    # ------------- Overfit Penalty -----
    kdtree = KDTree(points_gt)
    d, _ = kdtree.query(points)
    overfit = round(
        (len(np.where(d > cut_point)[0]) / len(points)) * 100
    )
    return recons, overfit


def load_image(path, dtype=torch.cuda.FloatTensor):
    img_pil, _ = get_image(path, -1)
    img_pil = crop_image(img_pil, 64)
    img_np = pil_to_np(img_pil)
    img_tensor = np_to_torch(img_np).type(dtype)
    img_np = np.moveaxis(img_np, 0, -1)
    return img_np, img_tensor


def images_mean(images, length=None):
    sum = np.zeros_like(images[0])
    for image in images:
        sum += image
    if length:
        return sum / length
    else:
        return sum / len(images)


def save_all_frames(hist, save_path):
    os.makedirs(save_path, exist_ok=True)
    hist_out = hist[0]
    digits = len(str(len(hist_out)))
    frames = []
    for i, out in enumerate(hist_out):
        out = cv2.normalize(
            out, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U
        )
        rgb_img = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        cv2.imwrite(
            f"{save_path}/frame-{str(i+1).zfill(digits)}.png", out
        )


def create_video(hist, save_path, fps=10):
    hist_out, hist_recons, hist_overfit = hist

    frames = []
    for i, out in enumerate(hist_out):
        out = cv2.normalize(
            out, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U
        )
        txt = f"iteration: {i}" #, {hist_recons[i]}, {hist_overfit[i]}"
        out = cv2.putText(
            out,
            str(txt),
            (5, 250),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        frames.append(out)
    print(f'saved to {save_path}')
    out = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*"MP4V"), fps, (256, 256)
    )
    for frame in frames:
        rgb_img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(rgb_img)
    out.release()
