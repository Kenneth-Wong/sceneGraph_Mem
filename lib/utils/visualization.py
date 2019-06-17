# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from six.moves import range
import PIL.Image as Image
from random import shuffle
import os.path as osp
import PIL.ImageColor as ImageColor
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
import cv2

from model.config import cfg

STANDARD_COLORS = [
    'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
    'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
]

NUM_COLORS = len(STANDARD_COLORS)
this_dir = osp.dirname(__file__)

try:
    FONT = ImageFont.truetype(osp.join(this_dir, '..', '..', 'data', 'helveticaneue.ttf'), 12)
except IOError:
    FONT = ImageFont.load_default()

try:
    FONT_BIG = ImageFont.truetype(osp.join(this_dir, '..', '..', 'data', 'helveticaneue.ttf'), 24)
except IOError:
    FONT_BIG = ImageFont.load_default()


def _draw_single_box(image, xmin, ymin, xmax, ymax, display_str, font, color='black', thickness=4):
    draw = ImageDraw.Draw(image)
    (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
    draw.line([(left, top), (left, bottom), (right, bottom),
               (right, top), (left, top)], width=thickness, fill=color)
    text_bottom = bottom
    # Reverse list and print from bottom to top.
    text_width, text_height = font.getsize(display_str)
    margin = np.ceil(0.05 * text_height)
    draw.rectangle(
        [(left, text_bottom - text_height - 2 * margin), (left + text_width,
                                                          text_bottom)],
        fill=color)
    draw.text(
        (left + margin, text_bottom - text_height - margin),
        display_str,
        fill='black',
        font=font)

    return image


def draw_bounding_boxes(image, gt_boxes, gt_labels):
    num_boxes = gt_boxes.shape[0]
    gt_boxes_new = gt_boxes.copy()
    gt_boxes_new = np.hstack((gt_boxes_new, gt_labels))
    disp_image = Image.fromarray(np.uint8(image[0]))

    for i in range(num_boxes):
        this_class = int(gt_boxes_new[i, 4])
        disp_image = _draw_single_box(disp_image,
                                      gt_boxes_new[i, 0],
                                      gt_boxes_new[i, 1],
                                      gt_boxes_new[i, 2],
                                      gt_boxes_new[i, 3],
                                      'N%02d-C%02d-%s' % (i, this_class, cfg.ind_to_class[this_class]),
                                      FONT,
                                      color=STANDARD_COLORS[this_class % NUM_COLORS])

    image[0, :] = np.array(disp_image)
    return image


def draw_gt_boxes(image, gt_boxes):
    num_boxes = gt_boxes.shape[0]
    disp_image = Image.fromarray(np.uint8(image[0]))

    list_gt = [i for i in range(num_boxes)]
    shuffle(list_gt)
    for i in list_gt:
        this_class = int(gt_boxes[i, 4])
        disp_image = _draw_single_box(disp_image,
                                      gt_boxes[i, 0],
                                      gt_boxes[i, 1],
                                      gt_boxes[i, 2],
                                      gt_boxes[i, 3],
                                      '%s' % (cfg.CLASSES[this_class]),
                                      FONT,
                                      color=STANDARD_COLORS[this_class % NUM_COLORS])

    new_image = np.empty_like(image)
    new_image[0, :] = np.array(disp_image)
    return new_image


def draw_predicted_boxes(image, scores, gt_boxes, labels=None):
    disp_image = Image.fromarray(np.uint8(image[0]))
    num_boxes = gt_boxes.shape[0]
    preds = np.argmax(scores, axis=1)
    if labels is None:
        labels = gt_boxes[:, 4]

    list_gt = [i for i in range(num_boxes)]
    shuffle(list_gt)
    for i in list_gt:
        this_class = int(labels[i])
        pred_class = preds[i]
        this_conf = scores[i, this_class]
        pred_conf = scores[i, pred_class]
        this_text = '%s|%.2f' % (cfg.CLASSES[this_class], this_conf)
        if this_class != pred_class:
            this_text += '(%s|%.2f)' % (cfg.CLASSES[pred_class], pred_conf)
        elif this_class == 0:
            this_text = '(X)'
        disp_image = _draw_single_box(disp_image,
                                      gt_boxes[i, 0],
                                      gt_boxes[i, 1],
                                      gt_boxes[i, 2],
                                      gt_boxes[i, 3],
                                      this_text,
                                      FONT,
                                      color=STANDARD_COLORS[this_class % NUM_COLORS])

    new_image = np.empty_like(image)
    new_image[0, :] = np.array(disp_image)
    return new_image


def draw_predicted_boxes_attend(image, scores, gt_boxes, attend, weight=None):
    disp_image = Image.fromarray(np.uint8(image[0]))
    num_boxes = gt_boxes.shape[0]
    preds = np.argmax(scores, axis=1)
    labels = gt_boxes[:, 4]

    list_gt = [i for i in range(num_boxes)]
    shuffle(list_gt)
    for i in list_gt:
        this_class = int(labels[i])
        pred_class = preds[i]
        this_conf = scores[i, this_class]
        pred_conf = scores[i, pred_class]
        this_text = '%.2f&' % attend[i, 0]
        if weight is not None:
            this_text += '%.2f&' % (weight[i] * num_boxes)
        this_text += '%s|%.2f' % (cfg.CLASSES[this_class], this_conf)
        if this_class != pred_class:
            this_text += '(%s|%.2f)' % (cfg.CLASSES[pred_class], pred_conf)
        elif this_class == 0:
            this_text = '(X)'
        disp_image = _draw_single_box(disp_image,
                                      gt_boxes[i, 0],
                                      gt_boxes[i, 1],
                                      gt_boxes[i, 2],
                                      gt_boxes[i, 3],
                                      this_text,
                                      FONT,
                                      color=STANDARD_COLORS[this_class % NUM_COLORS])

    new_image = np.empty_like(image)
    new_image[0, :] = np.array(disp_image)
    return new_image


def draw_predicted_boxes_test(image, scores, gt_boxes):
    disp_image = Image.fromarray(np.uint8(image[0] + cfg.PIXEL_MEANS))
    num_boxes = gt_boxes.shape[0]
    # Avoid background class
    preds = np.argmax(scores[:, 1:], axis=1) + 1
    wrong = False
    list_gt = [i for i in range(num_boxes)]
    shuffle(list_gt)
    for i in list_gt:
        this_class = int(gt_boxes[i, 4])
        pred_class = preds[i]
        this_conf = scores[i, this_class]
        pred_conf = scores[i, pred_class]
        if this_class != pred_class:
            this_text = '%s|%.2f' % (cfg.CLASSES[this_class], this_conf)
            this_text += '(%s|%.2f)' % (cfg.CLASSES[pred_class], pred_conf)
            wrong = True
        else:
            # this_text = '(X)'
            this_text = '%s|%.2f' % (cfg.CLASSES[this_class], this_conf)
        disp_image = _draw_single_box(disp_image,
                                      gt_boxes[i, 0],
                                      gt_boxes[i, 1],
                                      gt_boxes[i, 2],
                                      gt_boxes[i, 3],
                                      this_text,
                                      FONT,
                                      color=STANDARD_COLORS[this_class % NUM_COLORS])

    new_image = np.array(disp_image)
    return new_image, wrong


def draw_memory(mem, scale=1.0):
    # Set the boundary
    mem_image = np.minimum(np.mean(np.absolute(mem.squeeze(axis=0)), axis=2) * (255. / scale), 255.)
    # Just visualization
    mem_image = np.tile(np.expand_dims(mem_image, axis=2), [1, 1, 3])

    return mem_image[np.newaxis]


def draw_CAM(map, gt_image):
    weighted_map = np.uint8(np.minimum(map.squeeze(axis=0) * 255., 255.))
    # use opencv to get heat map, so we should swap the channel first
    base_image = gt_image[0].copy()
    h, w, _ = base_image.shape
    heatmap = cv2.applyColorMap(cv2.resize(weighted_map, (w, h)), cv2.COLORMAP_JET)
    result = heatmap * 0.3 + base_image * 0.5
    result = result[:, :, ::-1] # BGR to RGB, for show
    return result[np.newaxis].astype(np.float32)


def draw_weights(mem, scale=1.0):
    # Set the boundary
    mem_image = np.minimum(np.mean(np.absolute(mem.squeeze(axis=0)), axis=2) * (255. / scale), 255.)
    # Just visualization
    mem_image = np.tile(np.expand_dims(np.uint8(mem_image), axis=2), [1, 1, 3])

    return mem_image[np.newaxis]
