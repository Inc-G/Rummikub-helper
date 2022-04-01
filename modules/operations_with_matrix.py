"""
Performs operations with the matrix for the DPS algorithm.

The functions are:
- card_appears: checks if some copies of a card appear in a list
- remaining_cards_on_table: checks if there are cards on the table after removing the sets_taken
- choose_card: chooses the next card to look at
- from_index_to_set: given the matrix and an index, returns the corresponding valid set 
- sets_with_card: selects all sets where a card appears
- get_new_rows_and_cols_removed_or_decreased: updates the matrix
"""

import numpy as np
import pandas as pd

## Operations with the matrix


def card_appears(card:str, multiplicity:int, all_cards_taken:list)->bool:
    """
    Assumes all_cards_taken is sorted. Checks if there are multiplicity copies of card in all_cards_taken.
    """
    if len(all_cards_taken) == 0:
        return False
    index = 0
    card_multiplicity = multiplicity
    l = len(all_cards_taken)
    while card >= all_cards_taken[index]:
        if card == all_cards_taken[index]:
            card_multiplicity -=1
            if card_multiplicity == 0:
                return True
        index += 1
        if index>l-1:
            break
    return False
            
def remaining_cards_on_table(sets_taken:list, cards_on_table:dict, return_bool=False):
    """
    Checks if there are cards on the table after removing the sets_taken. If not return_bool,
    returns the cards remaining on the table (possibly the empty list), and if were taken cards from the hand.
    """
    all_cards_taken = sorted([card for good_set in sets_taken for card in good_set])
    cards_remaining = []
    counter = 0
    for card in cards_on_table.keys():
        counter += cards_on_table[card]
        all_taken = card_appears(card, cards_on_table[card], all_cards_taken)
        if not all_taken:
            cards_remaining.append(card)
    from_hand = len(all_cards_taken) > counter
    
    if return_bool:
        return len(cards_remaining)>0, from_hand
    else:
        return cards_remaining, from_hand
        
def choose_card(current_matrix:pd.DataFrame, sets_taken:list,
                rows_removed:list, columns_removed:list, cards_on_table:dict)->(bool, str):
    """
    Chooses the next card to look at. If there is a card which belongs to no sets returns True, card.
    Otherwise returns False, card that belongs to the least number of sets
    
    An input looks like (pd.df, [['2r','3b'],['3n','3o','3r']], [1,2], ['5b','7n'], {'2n':1,'3n':1, '7b':1})
    """
    new_matrix = current_matrix.drop(rows_removed).drop(columns_removed, axis=1)
    #print(new_matrix)
    multiplicities_cards = (new_matrix>0).sum(axis=0)
    
    remaining_cards, _ = remaining_cards_on_table(sets_taken, cards_on_table)
    #print('remaining_cards:', remaining_cards)
    
    min_, min_card = 9999999, '200r'
    for card in remaining_cards:
        c_val = multiplicities_cards[card]
        if c_val == 0:
            del new_matrix
            del multiplicities_cards
            return True, card
        if c_val < min_:
            min_ = c_val
            min_card = card
    
    del new_matrix
    del multiplicities_cards
    return False, min_card
        
def from_index_to_set(matrix:pd.DataFrame, index:int)->list:
    """
    returns the set corresponding to an index
    """
    row = matrix.loc[index,:]
    return list(row[row>0].index)

        
def sets_with_card(current_matrix, rows_removed, columns_removed, card)->list:
    """
    Returns all the sets containing a card.
    
    An input looks like (pd.DataFrame, [1,2], ['2r','3r'], '4o')
    """
    new_matrix = current_matrix.drop(rows_removed).drop(columns_removed, axis=1)
    card_column = new_matrix[card]
    choosen_index = card_column[card_column>0].index
    #print(new_matrix)
    valid_sets = []
    for index in choosen_index:
        valid_sets.append(from_index_to_set(new_matrix, index))
    del new_matrix 
    return valid_sets


def get_new_rows_and_cols_removed_or_decreased(current_matrix: pd.DataFrame,
                                               rows_removed: list,
                                               columns_removed: list,
                                               columns_decreased: list,
                                               valid_set: list):
    """
    returns which rows to drop, which columns to drop, and which indices to decrease
    """
    new_matrix = current_matrix.copy()
    new_matrix[columns_decreased] = new_matrix[columns_decreased]/2
    new_matrix = new_matrix.drop(rows_removed).drop(columns_removed, axis=1)
    
    indices_to_drop = set()
    dropped_cards = []
    decreased_cards = []

    for card in valid_set:
        card_column = new_matrix[card]
        sets_with_card = list(card_column[card_column>0].index)

        if card_column.max() == 1: #card appears with multiplicity 1
            indices_to_drop = indices_to_drop.union(set(sets_with_card))
            dropped_cards.append(card)
        else:
            decreased_cards.append(card)
            
    indices_to_drop = list(indices_to_drop)
    del new_matrix
    return indices_to_drop, dropped_cards, decreased_cards