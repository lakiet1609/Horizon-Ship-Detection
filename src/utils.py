import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image


def erode_img(image, kernel_size, iterations):
    _, binary_inverted_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    eroded_image = cv2.erode(binary_inverted_image, kernel, iterations=iterations)
    return eroded_image


def dilate_img(img, kernel_size, iterations):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    mask_image = cv2.morphologyEx(img, cv2.MORPH_DILATE, kernel, iterations=iterations)
    return mask_image


def draw_white_line(image, start_point, end_point, thickness):
    color = (255, 255, 255)
    cv2.line(image, start_point, end_point, color, thickness)
    return image


def check_area_of_mask(mask_image, min_width, min_height, expansion_size):
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask_image, connectivity=8)
    bounding_boxes = []
    for label in range(1, num_labels):
        x, y, w, h, _ = stats[label]
        if w >= min_width and h >= min_height:
            bbox = mask_image[y:y+h, x:x+w]
            white_pixels = np.sum(bbox == 255)

            x_expanded = max(x - expansion_size, 0)
            y_expanded = max(y - expansion_size, 0)
            w_expanded = min(w + 2 * expansion_size, mask_image.shape[1] - x_expanded)
            h_expanded = min(h + 2 * expansion_size, mask_image.shape[0] - y_expanded)
            
            bbox_expanded = mask_image[y_expanded:y_expanded+h_expanded, x_expanded:x_expanded+w_expanded]
            
            white_pixels_after = np.sum(bbox_expanded == 255)
            if white_pixels_after == white_pixels:
                bounding_boxes.append([x,y,w,h])
                cv2.rectangle(mask_image, (x, y), (x + w, y + h), (255, 255, 255), 2)
                
    return mask_image, bounding_boxes