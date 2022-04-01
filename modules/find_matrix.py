"""
Main function: from_cards_to_matrix. Takes a string of cards ['2r','4b',...] and returns the a matrix representing the varoius valid sets. This matrix has as many cols as the different types of cards, and as many rows as the different valid sets.
The jokers are distinct, so that each set does not contain multiple cards. row[i] corresponds to the valid set (a, b, c) iff matrix[i,a], matrix[i,b], matrix[i,c] != 0. If matrix[i,c] != 0, the number
matrix[i,c] is the multiplicity that card c appears (on table + hand).

Other functions: same_color_valid_sets and same_number_valid_sets.
Example for same_color_valid_sets:
({'r': ['13r', '3r', '4r', '2r', '1r'], 'b': ['4b', '3b']}, 0) --> 
{'r': [('2r', '3r', '4r'), ('1r', '2r', '3r'), ('1r', '2r', '3r', '4r')], 'b': []}
"""

import numpy as np
import itertools
import pandas as pd

import find_admissible_sets as admissible_sets

###########
## Auxiliary functions
###########

def create_dic_multiplicities(list_cards, diversify_jokers=False):
    dic = {}
    for card in list_cards:
        if dic.get(card)==None:
            dic[card] = 1
        else:
            dic[card] +=1
            
    if diversify_jokers:
        if dic.get('j') != None:
            numb_jokers = dic.pop('j')
        else:
            return dic
        if numb_jokers == 1:
            dic['jb'] = 1
        if numb_jokers == 2:
            dic['jb'] = 1
            dic['jr'] = 1
    return dic

def remove_empty_key_dic(dic):
    """
    Auxiliary
    """
    to_be_popped =[]
    for key in dic.keys():
        if len(dic[key])==0:
            to_be_popped.append(key)
    for key in to_be_popped:
        dic.pop(key)
    return dic

def add_color_to_single_tuple(my_tuple, color):
    """
    Auxiliary
    """
    res = []
    for number in my_tuple:
        if number[0] != 'j':
            res.append(number + color)
        else:
            res.append(number)
    return tuple(res)

def add_color_to_dic(dic):
    """
    Auxiliary
    """
    for key in dic.keys():
        new_value = []
        for valid_tuple in dic[key]:
            new_value.append(add_color_to_single_tuple(valid_tuple, key))
        dic[key] = new_value
    return dic

###########
## Organize present cards based on color and number
###########

def same_number_dict(present_cards):
    """
    Input: present_cards is a list of strings.
    Returns: Dictionary.
    
    Example: ['1b','2r','3b','13n','13r','2r'] --> {'13': ['13r', '13n'], '3': ['3b'], '2': ['2r'], '1': ['1b']}
    
    Remark: present_cards does not contain j, could contain repetitions; the values of the returned dic will not
    (see example).
    """
    present_cards = list(set(present_cards))
    number_sets = {}
    for card in present_cards:
        if len(card) == 3:
            number = card[:-1]
        else:
            number = card[0]
            
        if number_sets.get(number) == None:
            number_sets[number] = [card]
        else:
            number_sets[number].append(card)
    return number_sets
            
def same_color_dict(present_cards):
    """
    Input: present_cards is a list of strings.
    Returns: Dictionary.
    
    Example: ['1b','2r','3b','13n','13r','2r'] --> {'r': ['13r', '2r'], 'b': ['3b', '1b'], 'n': ['13n']}
    
    Remark: present_cards does not contain j, could contain repetitions; the values of the returned dic will not
    (see example).
    """
    present_cards = list(set(present_cards))
    color_sets = {}
    for card in present_cards:
        color = card[-1]
        
        if color_sets.get(color) == None:
            color_sets[color] = [card]
        else:
            color_sets[color].append(card)
    return color_sets

def same_color_valid_sets(same_col_dic, numb_jokers=0):
    """
    Input: same_col_dic is a dictionary with keys the colors and values a list of cards with that color
    (as in the output of same_color_dict).
    numb_jokers is an int.
    
    Uses the function valid_same_color_sets to return for each color the valid sets.
    Example: ({'r': ['13r', '3r', '4r', '2r', '1r'], 'b': ['4b', '3b']}, 0) -->
    {'r': [('2r', '3r', '4r'), ('1r', '2r', '3r'), ('1r', '2r', '3r', '4r')], 'b': []}
    """
    result = {}
    for color in same_col_dic.keys():
        cards = admissible_sets.drop_color_from_list(same_col_dic[color])
        valid_sets = admissible_sets.valid_same_color_sets(cards, numb_jokers)
        result[color] = valid_sets
    
    #add_color_to_dic transforms {'r': [('2', '3', '4')]} to {'r': [('2r', '3r', '4r')]}
    return add_color_to_dic(result)

def same_number_valid_sets(same_numb_dic, numb_jokers=0):
    """
    same_numb_dic is a dictionary with keys the numbers and values a list of cards with that number (as in the output
    of same_number_dict).
    numb_jokers is an int.
    
    Uses the function valid_same_number_sets to return for each number the valid sets.
    Example: ({'13': ['13r'], '4': ['4b', '4r', '4o'], '3': ['3b', '3r'], '2': ['2r'], '1': ['1r']}, 0) -->
    {'13': [], '4': [('4b', '4o', '4r')], '3': [], '2': [], '1': []}
    """
    result = {}
    for numb in same_numb_dic.keys():
        cards = same_numb_dic[numb]
        valid_sets = admissible_sets.valid_same_number_sets(cards, numb_jokers)
        result[numb] = valid_sets 
    return result
       
###########
## Get matrix
###########

def add_valid_tuples_from_dic(starting_matrix, new_dic, card_multiplicities, sorted_cards, numb_total_cards):
    """
    Auxiliary, used in get_matrix
    """
    result = []
    for key in new_dic.keys():
        for valid_set in new_dic[key]:
            new_row = np.zeros(numb_total_cards)
            for card in valid_set:
                occurences_card = card_multiplicities[card]
                new_row[sorted_cards.index(card)] = occurences_card
            result.append(new_row)
    return result
                
                

def get_matrix(card_multiplicities, number_valid_sets, color_valid_sets, return_pd_dataframe=False):
    """
    Input: return_pd_dataframe is a dic with keys cards and values their multiplicities.
    number_valid_sets and color_valid_sets are dict with values valid sets of cards
    return_pd_dataframe is a bool
    
    Returns: if return_pd_dataframe a pd.dataframe with columns the cards I have and indices the
    valid combinations of cards. The value at column '2r' and index a particular (valid) combination is
    0 if 2r does not appear in that combination and euqal to card_multiplicities['2r'] otherwise.
    For example, if the index corresponds to (3r, 4r, 5r) then the value on column 2r is 0, but on columns 3r,
    4r, 5r are all greater than 0    
    """
    sorted_cards = sorted(list(card_multiplicities.keys()))
    l = len(sorted_cards)
    
    result = add_valid_tuples_from_dic([], number_valid_sets, card_multiplicities, sorted_cards, l)
    result = result+add_valid_tuples_from_dic(result, color_valid_sets, card_multiplicities, sorted_cards, l)
    
    if return_pd_dataframe:
        return pd.DataFrame(result, columns=sorted_cards).drop_duplicates()
    else:
        return np.array(result)

# Main function

def from_cards_to_matrix(cards, print_intermediate_results=False, return_pd_dataframe=True):
    """
    Input: cards is a list strings, which are the cards I have.
    print_intermediate_results is a bool, = true for debug
    return_pd_dataframe is a bool
    
    Returns: if return_pd_dataframe a pd.dataframe with columns the cards I have and indices the
    valid combinations of cards. The value at column '2r' and index a particular (valid) combination is
    0 if 2r does not appear in that combination and euqal to card_multiplicities['2r'] otherwise.
    For example, if the index corresponds to (3r, 4r, 5r) then the value on column 2r is 0, but on columns 3r,
    4r, 5r are all greater than 0    
    """
    dic_mult = create_dic_multiplicities(cards)
    card_mult = dic_mult.copy()
    
    if print_intermediate_results:
        print('card multiplicities:', card_mult)
        
    if dic_mult.get('j')!=None:
        numb_jokers = dic_mult.pop('j')
    else:
        numb_jokers = 0
        
    if numb_jokers>0:
        card_mult.pop('j')
        if numb_jokers == 1:
            card_mult['jb'] = 1
        else:
            card_mult['jb'] = 1
            card_mult['jr'] = 1
    
    if print_intermediate_results:
        print('number of jokers:',numb_jokers)
        print('cards with jokers:', card_mult)

    same_number = same_number_dict(dic_mult)
    if print_intermediate_results:
        print('same number:', same_number)
        
    same_color = same_color_dict(dic_mult)
    if print_intermediate_results:
        print('same color:', same_color)

    same_color_set = remove_empty_key_dic(same_color_valid_sets(same_color, numb_jokers))
    if print_intermediate_results:
        print('same_color_set', same_color_set)
    
    same_number_set = remove_empty_key_dic(same_number_valid_sets(same_number, numb_jokers))
    if print_intermediate_results:
        print('same_number_set', same_number_set)
    
    return get_matrix(card_mult, same_number_set, same_color_set, return_pd_dataframe=return_pd_dataframe)