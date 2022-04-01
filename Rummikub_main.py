import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import os
import importlib

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)#avoids printing warnings when loading tf models

#Modules for obj detection/classification
from modules.neural_network_modules import get_info_photo as get_cards
from modules.neural_network_modules import model_number_and_color as my_models

# Solver
from modules import find_admissible_sets as admissible_sets
from modules import find_matrix as find_matrix
from modules import solver as solver

#importlib.reload(get_cards)
print('##############')
print('##############')
LOC_CARDS_ON_TABLE = input('Enter the location of the photo of the tiles on the table. For example, a valid input could be models/table_1.jpeg. ')
print('##############')
print('##############')
CARDS_ON_HAND = input('Enter the tiles in you hand separated by a comma and with no spaces. For example, 3b,2r,5n would mean 3 blue, 2 red, 5 black. ').split(',')
print('##############')
print('##############')

## Models

print('Loading classification models...')
model_color = my_models.load_model_color('models/weights_model_predict_color')
model_number = my_models.load_model_number('models/weights_model_predict_number/accuracy1.0')
print('Done!')

print('Loading object detection model...')
detect_fn = tf.saved_model.load('models/saved_model_obj_det')
print('Done!')


print('Processing cards on the table...')
cards_on_table_j = get_cards.get_cards_in_photo(LOC_CARDS_ON_TABLE, detect_fn, model_number, model_color,
                                                conf_threshold_number=.95, conf_threshold_color=.5,)
print('Done!')

def fix_jokers(list_of_cards):
    result = []
    for card in list_of_cards:
        if card[0] != 'j':
            if len(card)<4:
                result.append(card)
        else:
            result.append('j')
    return result

cards_on_table = fix_jokers(cards_on_table_j)        
cards_on_hand = fix_jokers(CARDS_ON_HAND)


matrix = find_matrix.from_cards_to_matrix(cards_on_table + cards_on_hand)
dic_cards_on_table = find_matrix.create_dic_multiplicities(cards_on_table, diversify_jokers=True)

result, winning_set = solver.solver(matrix, dic_cards_on_table)

if result:
    print('You can play! Here is how:')
    print(winning_set)
else:
    print("Looks like you can't play.")
