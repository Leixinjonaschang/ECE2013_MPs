#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import instances
from solve import solve


def get_pent_idx(pent):
    """
    Returns the index of a pentomino.
    """
    pidx = 0  # Index of pentomino
    for i in range(pent.shape[0]):  # Find first non-zero element
        for j in range(pent.shape[1]):  # Find first non-zero element
            if pent[i][j] != 0:  # Found non-zero element
                pidx = pent[i][j]  # Index of pentomino
                break  # Break out of inner loop
        if pidx != 0:  # Break out of outer loop
            break  # Break out of outer loop
    if pidx == 0:  # Pentomino not found
        return -1  # Pentomino not found
    return pidx - 1  # Pentomino found


def is_pentomino(pent, pents):  # pent is a numpy array
    """
    Checks if a pentomino pent is part of pents
    """
    pidx = get_pent_idx(pent)  # Index of pentomino
    if pidx == -1:  # Pentomino not found
        return False
    true_pent = pents[pidx]  # Pentomino found

    for flipnum in range(3):  # Check all flips
        p = np.copy(pent)  # Copy pentomino
        if flipnum > 0:  # Flip pentomino
            p = np.flip(pent, flipnum - 1)  # Flip pentomino
        for rot_num in range(4):  # Check all rotations
            if np.array_equal(true_pent, p):  # Check if pentominos are equal
                return True  # Pentomino found
            p = np.rot90(p)  # Rotate pentomino
    return False  # Pentomino not found


def add_pentomino(board, pent, coord, check_pent=False, valid_pents=None):  # pent is a numpy array
    """
    Adds a pentomino pent to the board. The pentomino will be placed such that
    coord[0] is the lowest row index of the pent and coord[1] is the lowest 
    column index. 
    
    check_pent will also check if the pentomino is part of the valid pentominos.
    """
    if check_pent and not is_pentomino(pent, valid_pents):  # Check if pentomino is valid
        return False  # Pentomino not valid
    for row in range(pent.shape[0]):  # Check if pentomino can be placed
        for col in range(pent.shape[1]):  # Check if pentomino can be placed
            if pent[row][col] != 0:  # Pentomino tile
                if board[coord[0] + row][coord[1] + col] != 0:  # Pentomino cannot be placed
                    return False  # Pentomino cannot be placed
                else:  # Pentomino can be placed
                    board[coord[0] + row][coord[1] + col] = pent[row][col]  # Place pentomino
    return True  # Pentomino can be placed


def remove_pentomino(board, pent_idx):
    # Remove pentomino from board
    board[board == pent_idx + 1] = 0  # Remove pentomino from board


def check_correctness(sol_list, board, pents):
    """
    Sol is a list of pentominos (possibly rotated) and their upper left coordinate
    """
    # All tiles used
    if len(sol_list) != len(pents):  # Check if all tiles used
        return False
    # Construct board
    sol_board = np.zeros(board.shape)  # Construct board
    seen_pents = [0] * len(pents)  # Construct seen_pents
    for pent, coord in sol_list:  # Check if pentomino can be placed
        pidx = get_pent_idx(pent)  # Index of pentomino
        if seen_pents[pidx] != 0:  # Check if pentomino has been seen
            return False  # Pentomino has been seen
        else:  # Pentomino has not been seen
            seen_pents[pidx] = 1  # Pentomino has been seen
        if not add_pentomino(sol_board, pent, coord, True, pents):  # Check if pentomino can be placed
            return False

    # Check same number of squares occupied
    if np.count_nonzero(board) != np.count_nonzero(sol_board):  # Check if same number of squares occupied
        return False
    # Check overlap
    if np.count_nonzero(board) != np.count_nonzero(np.multiply(board, sol_board)):  # Check if overlap
        return False

    return True


if __name__ == "__main__":
    """
    Run python Pentomino.py to check your solution. You can replace 'board' and 
    'pents' with boards of your own. You can start off easy with simple dominos.
    
    We won't guarantee which tests your code will be run on, however if it runs
    well on the pentomino set you should be fine. 
    """
    board = instances.board_6x10
    pents = instances.dominos
    sol_list = solve(board,
                     pents)  # sol_list is a list of pentominos (possibly rotated) and their upper left coordinate
    if check_correctness(sol_list, board, pents):  # Check if solution is correct
        print("PASSED!")  # Solution is correct
    else:
        print("FAILED...")  # Solution is incorrect
