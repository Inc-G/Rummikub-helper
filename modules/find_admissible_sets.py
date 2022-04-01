"""
Main functions: valid_same_color_sets and valid_same_number_sets. Find valid sets from a list of cards and an integer number of jokers. The jokers are distinct, so that each set does not contain multiple cards.

Examples:
valid_same_color_sets: ([1,2,4],1) --> [('1', '2', 'jb', '4'), ('2', 'jb', '4'), ('1', '2', 'jb')], and
valid_same_number_sets: (['4b','4n'], 1)-->[('4b', '4n', 'jb')]

Other function:
is_valid. Used to tell if a list of strings with value integers is a valid set (could be optimized!)
"""

import numpy as np
import itertools
import pandas as pd

###########
## Find admissible sets
# Find admissible sets with the same color
###########
def diversify_jokers(valid_set:list, numb_jokers=0):
    """
    Auxiliary. If valid_set contains no 'j', it does nothing.
    If it contains one 'j', it replaces it with 'jb'. Otherwise it replaces them with 'jb', 'jr'.
    """
    if numb_jokers==0:
        return [valid_set]
   
    without_j = [card for card in valid_set if card != 'j']
    if numb_jokers == 1:
        if len(valid_set) == len(without_j):
            return [valid_set]
        return [without_j + ['jb']]
    
    if numb_jokers == 2:
        if len(valid_set) == len(without_j):
            return [valid_set]
        if len(valid_set) == len(without_j) +1:
            return [without_j+['jb'], without_j+['jr']]
        return [without_j + ['jb','jr']]

    
def drop_color_from_list(my_list):
    """
    Input: my_list is a list of strings.
    Returns: list of integers
    
    Example: ['4r','13b','2r'] --> [4, 13, 2]
    """
    res = []
    for _ in my_list:
        if len(_) == 2:
            res.append(int(_[0]))
        else:
            res.append(int(_[:-1]))
    return res           


def which_numbers_are_present(my_list):
    """
    Input: my_list is a list of integers between 1 and 13.
    Returns: list of strings
    
    Example: [1, 2, 3, 13] --> ['1', '2', '3', 'j', 'j', 'j', 'j', 'j', 'j', 'j', 'j', 'j', '13']
    
    Returns a list of 'int' and 'j', which has 'int' at index j iff j+1 appears in my_list. 
    """
    result = ['j']*13
    for _ in my_list:
        result[int(_)-1] = str(_)
    return result


def is_valid(my_list, beginning, end, numb_jokers=0):
    """
    Input: my_list is a list of 'int' or 'j'.
    Returns: tuple (bool, int)
    
    Determines if the string my_list[beginning, end+1] is valid, possibly using some of the jokers.
    Example: for ['2','3','j','5'], beginning=0 and end=1 returns (T, 0) , if end=2 returns (F, 0).
    
    The returning tuple is as follows:
    - bool = number of 'int' from beginning to end (included) <= than numb_jokers.
    - int = numb_jokers - number of 'j' from beginning to end (included) 
    """
    index = beginning
    empty_spaces = 0
    while index <= end and empty_spaces <= numb_jokers:
        if my_list[index] == 'j':
            empty_spaces+=1
        index +=1
    return empty_spaces <= numb_jokers, numb_jokers-empty_spaces


def fill_with_jokers(valid_set, left_jokers):
    """
    Input: valid_set is a list and left_jokers an int.
    Returns: list of tuples.
    
    Given a valid set and the number of jokers left, returns all the sets where some of the cards
    have been replaced with the jokers.
    
    Example: (['1', '2','3'], 1) --> [('1', '2', '3'), ('1', '2', 'j'), ('1', '3', 'j'), ('2', '3', 'j')]
    """
    result = [tuple(sorted(valid_set))]
    #sorted is there so that in the next function i can use list(set()) to avoid repetitions
    
    l = len(valid_set)
    for numb_jokers in range(1, left_jokers+1):
        for new_j in list(itertools.combinations(valid_set, l-numb_jokers)):
            result.append(tuple(sorted(list(new_j)+['j']*numb_jokers)))
            #sorted is there so that in the next function i can use list(set()) to avoid repetitions
    return result

#Main function
def valid_same_color_sets(my_list, numb_jokers=0):
    """
    Input: my_list is a list of integers between 1 and 13.
    numb_jokers is an integer representing the number of jokers.
    Returns: list of tuples
    
    my_list represents which red cards I am considering. For example, [1,2,4] means I am considering
    the cards 1r, 2r, 4r.
    Returns a list of tuples of all the combinations that are valid.
    
    Example: ([1,2,4],1) --> [('1', '2', 'j', '4'), ('2', 'j', '4'), ('1', '2', 'j')]
    """
    enc_list = which_numbers_are_present(my_list)
    result = []
    for _ in range(13):
        end = _+2
        while end <= 12:
            valid_string, jokers_left = is_valid(enc_list, _, end, numb_jokers)
            if valid_string:
                result+= fill_with_jokers(enc_list[_: end+1], jokers_left)
            else: 
                break
            end+=1
    result = list(set(result))
    
    #diversify the jokers
    different_j = []
    for val_set in result:
        different_j+=diversify_jokers(val_set, numb_jokers)
    return different_j
            
###########
# Find admissible sets with the same number
###########

#Main function
def valid_same_number_sets(colors_present, numb_jokers=0):
    """
    Input: Colors present is a list of strings. numb_jokers is an int.
    Returns a list of tuples.
    
    Returns the valid sets with the same number.
    Example: (['4b','4n'], 1)-->[('4b', '4n', 'j')]
    """
    cards = colors_present + ['j']*numb_jokers
    cards = sorted(cards)
    valid_sets = list(itertools.combinations(cards, 3)) + list(itertools.combinations(cards, 4))
    valid_sets = list(set(valid_sets))
    
    #diversify the jokers
    different_j = []
    for val_set in valid_sets:
        different_j+=diversify_jokers(val_set, numb_jokers)
    return different_j


