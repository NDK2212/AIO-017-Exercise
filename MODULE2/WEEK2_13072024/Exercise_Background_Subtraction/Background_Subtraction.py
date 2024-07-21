import math
import numpy as np
import cv2


def compute_difference(bg, ob):
    diff = cv2.absdiff(bg, ob)
    return diff


def compute_binary_mask(diff):
    _, thres = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY)
    cv2.imshow('thres', thres)
    return thres


def replace_background(bg1_image, bg2_image, ob_image):
    diff = compute_difference(ob_image, bg1_image)
    dif_binary = compute_binary_mask(diff)
    output = np.where(dif_binary == 0, bg2_image, ob_image)
    return output


if __name__ == "__main__":
    bg1_image = cv2.imread(
        'MODULE2\WEEK2_13072024\Exercise_Background_Subtraction\GreenBackground.png', 1)
    bg1_image = cv2.resize(bg1_image, (678, 381))

    ob_image = cv2.imread(
        'MODULE2\WEEK2_13072024\Exercise_Background_Subtraction\Object.png', 1)
    ob_image = cv2.resize(ob_image, (678, 381))

    bg2_image = cv2.imread(
        'MODULE2\WEEK2_13072024\Exercise_Background_Subtraction\AfterBackground.jpg', 1)
    bg2_image = cv2.resize(bg2_image, (678, 381))

    output = replace_background(bg1_image, bg2_image, ob_image)
    cv2.imshow('Output', output)
    cv2.waitKey(0)
