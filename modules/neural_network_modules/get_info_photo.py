"""
Main function: get_cards_in_photo. Using detect_fn detects where the tiles in the image at image_path are. Using
model_number, model_color, for each tile detected, gets its number and color. 

Needs tensorflow.
"""

import numpy as np
import tensorflow as tf

import os
import matplotlib.pyplot as plt
from PIL import Image


NUMB_CARDS = ['1', '10', '11', '12', '13', '2', '3', '4', '5', '6', '7', '8', '9', 'j']
COL_CARDS = ['b', 'n', 'o', 'r']

#############
## Get cards in photo
############

def get_image(image, from_path):
    if from_path:
        return np.array(Image.open(image))
    else:
        return np.array(image)

def export_batch_of_cards(card_boxes, image_np, image_shape):
    """
    card_boxes are the boxes detected by the obj detection model, image_np is the imput image of shape image_shape
    
    returns a list with image_np cropped along each box in card_boxes
    """
    all_cards = []
    for _ in card_boxes:
        height, width, channel = image_shape
        ymin = int((_[0]*height))
        xmin = int((_[1]*width))
        ymax = int((_[2]*height))
        xmax = int((_[3]*width))
        result = np.array(image_np[ymin:ymax,xmin:xmax]) 
        all_cards.append(result)
    return all_cards


def get_cards_from_photo(image, confidence_threshold, detect_fn, from_path):
    """
    Returns a list of photos of cards in the image
    """
    image_np = get_image(image, from_path)
    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = input_tensor[tf.newaxis, ...]

    new_detections = detect_fn(input_tensor)

    card_boxes = new_detections['detection_boxes'][0][:].numpy()[new_detections['detection_scores'][0].numpy() > confidence_threshold]
    
    all_cards = export_batch_of_cards(card_boxes,image_np, image_np.shape)
    return all_cards


def get_info_card(image, model, detect_number=False, confidence_threshold=None, accept_input=True):
    """
    Gets number and color from a card. 
    """
    if detect_number:
        new_shape = (96, 96)
    else:
        new_shape = (20, 20)
    image_r = tf.image.resize(image, new_shape)
    prediction = model(image_r[np.newaxis, ...]).numpy()[0]
    
    if confidence_threshold is not None:
        if np.max(prediction) < confidence_threshold and accept_input:
            if detect_number:
                print('The model predicts ', NUMB_CARDS[np.argmax(prediction)], ' with confidence ', np.max(prediction),'. As it is below the set threshold, we need to confirm it.')
            else:
                print('The model predicts ', COL_CARDS[np.argmax(prediction)], ' with confidence ', np.max(prediction),'. As it is below the set threshold, we need to confirm it.')
                
            plt.imshow(image)
            plt.show()
            if detect_number:
                val = input('Which number is it? If this is not a card write 123. ')
            else:
                val = input('Which color is it? The colors are b, n, o, r. If this is not a card write 123. ')
            return val
    
        if detect_number:
            return NUMB_CARDS[np.argmax(prediction)]
        else:
            return COL_CARDS[np.argmax(prediction)]
    
    if detect_number:
        return NUMB_CARDS[np.argmax(prediction)]
    else:
        return COL_CARDS[np.argmax(prediction)]


def get_cards_in_photo(image, detect_fn, model_number, model_color,
                       conf_threshold_color=.6, conf_threshold_number=.6,
                       conf_treshold_bounding_box=.985, accept_input=True, from_path=True):
    """
    Returns a list with entries str(card number) + str(card color) for every card detected by
    the object detection function detect_fn.
    The information about the card is obtained using model_number, model_color.
    If the confidence on the number is less than conf_threshold_detect_numb and accept_input==True you have to
    enter which card is detected.
    conf_treshold_bounding_box is used to keep all the bounding boxes which have probability>=conf_treshold_bounding_box
    to be a card
    """
    all_cards = get_cards_from_photo(image, conf_treshold_bounding_box, detect_fn, from_path)
    result = []
    for card in all_cards:
        number = get_info_card(card, model_number,  True,
                               confidence_threshold=conf_threshold_number, accept_input=accept_input)
        color = get_info_card(card, model_color, confidence_threshold=conf_threshold_color, accept_input=accept_input)
        result.append(number+color)
    return result
