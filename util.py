import numpy as np
import cv2
import scipy.io as sio
import random


def add_gaussian_noise(img, model_path, sigma):
    index = model_path.rfind("/")
    if sigma > 0:
        noise = np.random.normal(scale=sigma / 255., size=img.shape).astype(np.float32)
        sio.savemat(model_path[0:index] + '/noise.mat', {'noise': noise})
        noisy_img = (img + noise).astype(np.float32)
    else:
        noisy_img = img.astype(np.float32)
    cv2.imwrite(model_path[0:index] + '/noisy.png',
                np.squeeze(np.int32(np.clip(noisy_img, 0, 1) * 255.)))
    return noisy_img


def load_np_image(path, is_scale=True):
    img = cv2.imread(path, -1)
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    img = np.expand_dims(img, axis=0)
    if is_scale:
        img = np.array(img).astype(np.float32) / 255.
    return img


def mask_pixel(img, model_path, rate, attention_map=None):
    index = model_path.rfind("/")
    masked_img = img.copy()
    mask = np.ones_like(masked_img)

    h, w = img.shape[1:3]
    possible_mask_indices = [(x, y) for x in range(h) for y in range(w) 
                             if attention_map is None or attention_map[0, x, y, 0] == 0]
    num_to_mask = int(len(possible_mask_indices) * rate)

    mask_indices = random.sample(possible_mask_indices, num_to_mask)
    for x, y in mask_indices:
        masked_img[:, x, y, :] = 0
        mask[:, x, y, :] = 0

    cv2.imwrite(model_path[0:index] + '/masked_img.png', np.squeeze(np.uint8(np.clip(masked_img, 0, 1) * 255.)))
    cv2.imwrite(model_path[0:index] + '/mask.png', np.squeeze(np.uint8(np.clip(mask, 0, 1) * 255.)))

    return masked_img, mask


