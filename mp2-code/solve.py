# -*- coding: utf-8 -*-
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import time


def solve(board, pents):
    """
    This is the function you will implement. It will take in a numpy array of the board
    as well as a list of n tiles in the form of numpy arrays. The solution returned
    is of the form [(p1, (row1, col1))...(pn,  (rown, coln))]
    where pi is a tile (may be rotated or flipped), and (rowi, coli) is
    the coordinate of the upper left corner of pi in the board (lowest row and column index
    that the tile covers).

    -Use np.flip and np.rot90 to manipulate pentominos.

    -You may assume there will always be a solution.
    """
    board = np.array(board)
    # change the board for better processing
    board[board == 0] = -1
    board[board == 1] = 0

    pents_dict = {}
    for p in pents:
        pents_dict[label_pent(p)] = get_variants(p)

    start_time = time.time()
    # get the result board first
    _, res_board = recursive_dfs(board, dict(pents_dict))
    end_time = time.time()
    print(f"The result board with pent labels is: \n {res_board}")
    print(f"The time used to find the result board with recursive DFS is: {end_time - start_time}")

    visualize_result_board(res_board)  # visualize the result board

    # produce the final solution
    pents_coords = defaultdict(list)
    for x, y in np.ndindex(res_board.shape):
        if res_board[x][y] in pents_dict:
            pents_coords[res_board[x][y]].append((x, y))

    solution = []

    for plabel, coords in pents_coords.items():
        min_x, max_x, min_y, max_y = coords[0][0], coords[0][0], coords[0][1], coords[0][1]
        for coord in coords:
            min_x, max_x = min(min_x, coord[0]), max(max_x, coord[0])
            min_y, max_y = min(min_y, coord[1]), max(max_y, coord[1])
        # create a piece with the minimum size for the pent
        piece = np.zeros((max_x - min_x + 1, max_y - min_y + 1))
        for coord in coords:
            piece[coord[0] - min_x][coord[1] - min_y] = plabel
        solution.append((piece.astype(int), (min_x, min_y)))

    return solution


def recursive_dfs(board, pents_dict):
    # recursively call the function to find the solution with depth first search
    if not pents_dict:  # if all the pents are placed, return the result
        return True, board  # return True and the result board, which means "found"

    board_shape = board.shape

    for x, y in ((x, y) for y in range(board_shape[1]) for x in range(board_shape[0])):
        # find the the first unfilled position to add a pent
        if board[x][y] == 0:
            for p, variants in pents_dict.items():
                # try all the variants of a pent
                for variant in variants:
                    to_continue = False

                    # check if it's possible to add the pent on the board
                    for x_move, y_move in variant:
                        if not in_bound_test(board_shape, x + x_move, y + y_move) or board[x + x_move][y + y_move] != 0:
                            # if the pent is out of bound or overlaps with other pents, try the next variant
                            to_continue = True
                            break  # break if we cannot add a new pent
                    # continue to the next loop if we cannot add a new pent
                    if to_continue:
                        continue

                    # create a new board and a new pents_map for recursion
                    new_board = np.array(board)

                    new_pents_map = dict(pents_dict)
                    new_pents_map.pop(p)

                    # add the pent to the board
                    for x_move, y_move in variant:
                        new_board[x + x_move][y + y_move] = p

                    # recursively running
                    found, res_board = recursive_dfs(new_board, new_pents_map)

                    if found:
                        return True, res_board

            return False, board


def in_bound_test(board_shape, x, y):  # check if a coordinate is in bound
    return 0 <= x < board_shape[0] and 0 <= y < board_shape[1]


def get_variants(pentomino):
    # get all the variants of a pentomino, like flipping and rotating
    variants = set()  # use a set to remove duplicates
    for flip_count in range(3):
        flipped_pentomino = np.copy(pentomino)
        if flip_count > 0:
            flipped_pentomino = np.flip(pentomino, flip_count - 1)
        for rotation_count in range(4):
            rotated_pentomino = np.rot90(flipped_pentomino, rotation_count)
            variants.add(reformulate_variant(rotated_pentomino))

    return list(variants)


def reformulate_variant(variant):
    # reformulate the variant to a tuple of coordinates
    coords = []
    for x, y in np.ndindex(variant.shape):
        if variant[x][y] != 0:
            coords.append((x, y))

    origin = coords[0]

    for coord in coords:
        if coord[1] < origin[1]:
            origin = coord
        elif coord[1] == origin[1] and coord[0] < origin[0]:
            origin = coord

    variant = tuple()
    for coord in coords:
        variant += ((coord[0] - origin[0], coord[1] - origin[1]),)

    return tuple(sorted(variant))


def label_pent(pent):
    for i in range(pent.shape[0]):
        for j in range(pent.shape[1]):
            if pent[i][j] != 0:
                plabel = pent[i][j]
                break
        if plabel != 0:
            break
    return plabel


def visualize_result_board(res_board):
    # color the board for better visualization
    color_board = np.zeros(res_board.shape)
    max_element = np.max(res_board)
    for x, y in np.ndindex(res_board.shape):
        if res_board[x][y] != 0:
            color_board[x][y] = res_board[x][y] / (max_element + 1)

    # plot the colored board for 2 seconds and close the plot
    plt.imshow(color_board, cmap='viridis', interpolation='nearest')
    plt.show(block=False)  # set block=False to make the plot not block the code to proceed
    plt.title("The visualization of the result board")
    plt.pause(3)
    plt.close()
