"""
Main function: solver.
"""


import numpy as np
import itertools
import pandas as pd

import importlib
import find_admissible_sets as admissible_sets
import find_matrix as find_matrix
import operations_with_matrix as operations


def solver(current_matrix:pd.DataFrame, cards_on_table:dict, sets_taken=[], rows_removed=[], columns_removed=[],
           columns_decreased=[], print_intermediate_outputs=False):
    """
    current_matrix is a pd.df with columns the cards, rows the admissible sets. The jokers are distinct, so
    that each set does not contain multiple cards. So cards_on_table does not have a 'j' key, if it has a joker
    it either has 'jb' as key or 'jb', 'jr' as keys.
    
    Checks if there is a set of admissible sets that forms a partition of a subset of all the cards (table + hand)
    which contains all the cards on the table. If there is such a set, it returns True, such a set. Otherwise
    False, []
    
    Step 1: check if there are still cards on the table. If not, we check if we took cards from the hand or if
    we can take cards from hand. 
    
    Step 2: pick a card on the table that belongs to the least number of sets. If this number is 0, end. Otherwise
    
    Step 3: consider all the sets containing the card from step 2. If we can win, one of these sets
    must be taken. 
    
    Step 4: For each one of the sets of step 3, assume you took it. This will give a new matrix, new cards on table,..
    For each of these new combinations, check if you win. If there is a winning combination stop and return it,
    if not return false, []
    """
    
    ## table is empty
    cards_remaining, taken_from_hand = operations.remaining_cards_on_table(sets_taken,
                                                                cards_on_table,
                                                                return_bool=True)
    if not cards_remaining: 
        if taken_from_hand:
            return True, sets_taken
        return len(rows_removed)< current_matrix.shape[0], sets_taken
    
    ## table is not empty
    # we check if we already lost (i.e. if there is a card belonging to no valid set). If not, we choose a
    # card belonging to the least number of valid sets (i.e. next_card)
    
    already_lost, next_card = operations.choose_card(current_matrix,
                                          sets_taken,
                                          rows_removed,
                                          columns_removed,
                                          cards_on_table)
    
    if already_lost:
        return False, []
    if print_intermediate_outputs:
        print('next_card:', next_card)
    
    # list of set containing next_card
    current_valid_sets = operations.sets_with_card(current_matrix, rows_removed, columns_removed, next_card)
    
    for valid_set in current_valid_sets:
        new_rows_removed, new_col_removed, new_col_decreased = operations.get_new_rows_and_cols_removed_or_decreased(current_matrix,
                                                                                                          rows_removed,
                                                                                                          columns_removed,
                                                                                                          columns_decreased,
                                                                                                          valid_set)
        if print_intermediate_outputs:
            print('valid_set', valid_set)
            print('prev col removed', columns_removed)
            print('new col_removed', new_col_removed)
            print('previous col decreased', columns_decreased)
            print('new col_decreased', new_col_decreased)
            new_matrix = current_matrix.copy()
            new_matrix[columns_decreased + new_col_decreased] = new_matrix[columns_decreased + new_col_decreased]/2
            new_matrix = current_matrix.drop(rows_removed + new_rows_removed).drop(columns_removed + new_col_removed, axis=1)
            print('new matrix:')
            print('new matrix cols:', new_matrix.columns)
            print(new_matrix)
        
        finished, winning_set = solver(current_matrix,
                                       cards_on_table,
                                       sets_taken + [valid_set],
                                       rows_removed + new_rows_removed,
                                       columns_removed + new_col_removed,
                                       columns_decreased + new_col_decreased)
        if print_intermediate_outputs:
            print('finished:', finished)
            print('-------')
        
        if finished:
            return True, winning_set
    return False, []