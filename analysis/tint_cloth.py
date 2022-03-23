
"""
This script provides functions for tinting and normalizing image observations.
These operations are done to make the images more closely resemble the images
encountered during our physical robot experiments.
"""

import numpy as np
import cv2
import os
from skimage import color
from skimage import img_as_float


minimum = 20
maximum = 235

minimum_new = 60
maximum_new = 255

newRange = maximum_new - minimum_new
oldRange = maximum - minimum

def normalize(img_crop):
    """
    This function blurs the images as well as normalizing them.
    """
    img_crop = np.where(img_crop > minimum, img_crop, minimum)
    img_crop = np.where(img_crop < maximum, img_crop, maximum)
    var = newRange/oldRange
    img_crop = (img_crop - minimum) * var + minimum_new
    img_crop = img_crop.astype(np.uint8)
    img_crop = cv2.cvtColor(img_crop[:, :, 0], cv2.COLOR_GRAY2BGR)
    img_crop = cv2.fastNlMeansDenoisingColored(img_crop, None, 7, 7, 7, 21)
    
    img_crop = img_crop[:, :, 0]
    img_crop = np.transpose(np.array([img_crop, img_crop, img_crop]), (1, 2, 0))
    
    return img_crop



def tint_cloth(img, light_multiplier, dark_multiplier, background_multiplier):
    img = remove_border(img)
    light_mask = mask_light_blue(img)
    dark_mask = mask_dark_blue(img)
    full_mask = mask_full(img)

    img_cp = img.copy()
    img_cp = np.array([img_cp[:, :, 0], img_cp[:, :, 1], img_cp[:, :, 2]])
    img_cp = np.transpose(img_cp, [1,2,0])

    light = cv2.bitwise_and(img_cp, img_cp, mask=light_mask)
    gray_light = cv2.cvtColor(light, cv2.COLOR_BGR2GRAY)
    grayscale_light = img_as_float(gray_light)
    light_image = color.gray2rgb(grayscale_light)
    tint_light_parts = light_image * light_multiplier
    tint_light_parts = tint_light_parts * 255

    dark = cv2.bitwise_and(img_cp, img_cp, mask=dark_mask)
    gray_dark = cv2.cvtColor(dark, cv2.COLOR_BGR2GRAY)
    grayscale_dark = img_as_float(gray_dark)
    dark_image = color.gray2rgb(grayscale_dark)
    tint_dark_parts = dark_image * dark_multiplier
    tint_dark_parts = tint_dark_parts * 255

    tinted_cloth = cv2.bitwise_or(tint_light_parts, tint_dark_parts)
    tinted_cloth = tinted_cloth.astype(np.uint8)

    background = cv2.bitwise_and(img, img, mask=np.invert(full_mask))
    background_tint = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
    grayscale_background = img_as_float(background_tint)
    background_img = color.gray2rgb(grayscale_background)
    tint_background = background_img * background_multiplier
    tint_background = tint_background * 255
    tint_background = tint_background.astype(np.uint8)
    
    final_img = cv2.bitwise_or(tinted_cloth, tint_background)
    gray_img = cv2.cvtColor(final_img, cv2.COLOR_BGR2GRAY)
    gray_img = np.stack((gray_img, gray_img, gray_img))
    gray_img = np.transpose(gray_img, (1,2,0))
    return gray_img

def mask_light_blue(img):
    # R: 75, G: 149, B: 255
    light_mask = np.zeros(img.shape[:2], dtype="uint8")
    light_mask[img[:, :, 2] < 85] = 255
    img_mask1 = cv2.bitwise_and(img, img, mask=light_mask)
    light_mask[img_mask1[:, :, 0] < 65] = 0
    img_mask1 = cv2.bitwise_and(img, img, mask=light_mask)
    light_mask[img_mask1[:, :, 1] < 70] = 0
    return light_mask

def mask_dark_blue(img):
    # R: 75, G: 63, B: 205
    dark_mask = np.zeros(img.shape[:2], dtype="uint8")
    dark_mask[img[:, :, 2] < 85] = 255
    img_mask1 = cv2.bitwise_and(img, img, mask=dark_mask)
    dark_mask[img_mask1[:, :, 0] < 65] = 0
    img_mask1 = cv2.bitwise_and(img, img, mask=dark_mask)
    dark_mask[img_mask1[:, :, 1] > 67] = 0
    dark_mask[img_mask1[:, :, 1] < 50] = 0
    return dark_mask

def mask_full(img):
    light_mask = np.zeros(img.shape[:2], dtype="uint8")
    light_mask[img[:, :, 2] < 85] = 255
    img_mask1 = cv2.bitwise_and(img, img, mask=light_mask)
    light_mask[img_mask1[:, :, 0] < 65] = 0
    return light_mask

def threshold_border(border):
    mask = mask_full(border)
    border = cv2.bitwise_and(border, border, mask=mask)
    border_white = (np.ones(border.shape) * 255).astype("uint8")
    border_white = cv2.bitwise_and(border_white, border_white, mask=np.invert(mask))
    border = cv2.bitwise_or(border, border_white)
    return border

def remove_border(img):
    img_cp = img.copy()

    top_border = img_cp[:29, :, :]
    top_border = threshold_border(top_border)
    # canny_cut_image_horizontal(top_border)
    bottom_border = img_cp[195:, :, :]
    bottom_border = threshold_border(bottom_border)
    # canny_cut_image_horizontal(bottom_border)
    left_border = img_cp[29:195, :29, :]
    left_border = threshold_border(left_border)
    # canny_cut_image_vertical(left_border)
    right_border = img_cp[29:195, 195:, :]
    right_border = threshold_border(right_border)
    # canny_cut_image_vertical(right_border)

    img_cp[:29, :, :] = top_border
    img_cp[195:, :, :] = bottom_border
    img_cp[29:195, :29, :] = left_border
    img_cp[29:195, 195:, :] = right_border
    return img_cp

if __name__ == '__main__':
    img_folder_path = "test_17x17_train_25x25"
    light_brown_multiplier = [0.5, 0.55, 0.85] #BGR
    dark_brown_multiplier = [0.5, 0.5, 0.6] #BGR

    results_folder = "results"
    if os.path.exists(results_folder):
        remove_command = 'rm -r ' + results_folder
        os.system(remove_command)
    os.mkdir(results_folder)

    for file in np.sort(os.listdir(img_folder_path)):
        print(file)
        img = cv2.imread(os.path.join(img_folder_path, file))
        # remove_border(img)
        final_img = tint_cloth(img, light_brown_multiplier, dark_brown_multiplier)
        cv2.imwrite('%s/%s'%(results_folder, file), final_img)
