from time import sleep
from math import inf
from random import randint, choice
import matplotlib.pyplot as plt


class ultimateTicTacToe:
    def __init__(self):
        """
        Initialization of the game.
        """
        self.board = [['_', '_', '_', '_', '_', '_', '_', '_', '_'],
                      ['_', '_', '_', '_', '_', '_', '_', '_', '_'],
                      ['_', '_', '_', '_', '_', '_', '_', '_', '_'],
                      ['_', '_', '_', '_', '_', '_', '_', '_', '_'],
                      ['_', '_', '_', '_', '_', '_', '_', '_', '_'],
                      ['_', '_', '_', '_', '_', '_', '_', '_', '_'],
                      ['_', '_', '_', '_', '_', '_', '_', '_', '_'],
                      ['_', '_', '_', '_', '_', '_', '_', '_', '_'],
                      ['_', '_', '_', '_', '_', '_', '_', '_', '_']]
        self.maxPlayer = 'X'
        self.minPlayer = 'O'
        self.maxDepth = 3
        # The start indexes of each local board
        self.globalIdx = [(0, 0), (0, 3), (0, 6), (3, 0), (3, 3), (3, 6), (6, 0), (6, 3), (6, 6)]

        # Start local board index for reflex agent playing
        self.startBoardIdx = 4
        # self.startBoardIdx=randint(0,8)

        # utility value for reflex offensive and reflex defensive agents
        self.winnerMaxUtility = 10000
        self.twoInARowMaxUtility = 500
        self.preventThreeInARowMaxUtility = 100
        self.cornerMaxUtility = 30

        self.winnerMinUtility = -10000
        self.twoInARowMinUtility = -100
        self.preventThreeInARowMinUtility = -500
        self.cornerMinUtility = -30

        self.expandedNodes = 0
        self.currPlayer = True

    def printGameBoard(self):
        """
        This function prints the current game board.
        """
        print('\n'.join([' '.join([str(cell) for cell in row]) for row in self.board[:3]]) + '\n')
        print('\n'.join([' '.join([str(cell) for cell in row]) for row in self.board[3:6]]) + '\n')
        print('\n'.join([' '.join([str(cell) for cell in row]) for row in self.board[6:9]]) + '\n')

    def checkBlocked(self, isMax, start):
        i = start[0]
        j = start[1]
        p = 0

        if isMax:
            player = self.maxPlayer
            opponent = self.minPlayer
        else:
            player = self.minPlayer
            opponent = self.maxPlayer

        b = self.board
        for row in range(3):
            if b[row + i][j] == b[row + i][j + 1] and b[row + i][j] == opponent and b[row + i][j + 2] == player:
                p += 1
            if b[row + i][j] == b[row + i][j + 2] and b[row + i][j] == opponent and b[row + i][j + 1] == player:
                p += 1
            if b[row + i][j + 1] == b[row + i][j + 2] and b[row + i][j + 1] == opponent and b[row + i][j] == player:
                p += 1

        for col in range(3):
            if b[i][col + j] == b[i + 1][col + j] and b[i][col + j] == opponent and b[i + 2][col + j] == player:
                p += 1
            if b[i][col + j] == b[i + 2][col + j] and b[i][col + j] == opponent and b[i + 1][col + j] == player:
                p += 1
            if b[i + 1][col + j] == b[i + 2][col + j] and b[i + 1][col + j] == opponent and b[i][col + j] == player:
                p += 1

        # Unblocked diagonals
        if b[i][j] == b[i + 1][j + 1] and b[i][j] == opponent and b[i + 2][j + 2] == player:
            p += 1
        if b[i][j] == b[i + 2][j + 2] and b[i][j] == opponent and b[i + 1][j + 1] == player:
            p += 1
        if b[i + 1][j + 1] == b[i + 2][j + 2] and b[i + 1][j + 1] == opponent and b[i][j] == player:
            p += 1
        # --------------------------------------------------------------------------
        if b[i + 2][j] == b[i + 1][j + 1] and b[i + 2][j] == opponent and b[i][j + 2] == player:
            p += 1
        if b[i + 2][j] == b[i][j + 2] and b[i + 2][j] == opponent and b[i + 1][j + 1] == player:
            p += 1
        if b[i + 1][j + 1] == b[i][j + 2] and b[i + 1][j + 1] == opponent and b[i + 2][j] == player:
            p += 1

        return p

    def checkUnblocked(self, isMax, start):
        i = start[0]
        j = start[1]
        p = 0

        if isMax:
            player = self.maxPlayer
        else:
            player = self.minPlayer

        b = self.board
        # Unblocked rows
        for row in range(3):
            if b[row + i][j] == b[row + i][j + 1] and b[row + i][j] == player and b[row + i][j + 2] == '_':
                p += 1
            if b[row + i][j] == b[row + i][j + 2] and b[row + i][j] == player and b[row + i][j + 1] == '_':
                p += 1
            if b[row + i][j + 1] == b[row][j + 2] and b[row + i][j + 1] == player and b[row + i][j] == '_':
                p += 1

        # Unblocked columns
        for col in range(3):
            if b[i][col + j] == b[i + 1][col + j] and b[i][col + j] == player and b[i + 2][col + j] == '_':
                p += 1
            if b[i][col + j] == b[i + 2][col + j] and b[i][col + j] == player and b[i + 1][col + j] == '_':
                p += 1
            if b[i + 1][col + j] == b[i + 2][col] and b[i + 1][col + j] == player and b[i][col + j] == '_':
                p += 1

        # Unblocked diagonals
        if b[i][j] == b[i + 1][j + 1] and b[i][j] == player and b[i + 2][j + 2] == '_':
            p += 1
        if b[i][j] == b[i + 2][j + 2] and b[i][j] == player and b[i + 1][j + 1] == '_':
            p += 1
        if b[i + 1][j + 1] == b[i + 2][j + 2] and b[i + 1][j + 1] == player and b[i][j] == '_':
            p += 1
        # --------------------------------------------------------------------------
        if b[i + 2][j] == b[i + 1][j + 1] and b[i + 2][j] == player and b[i][j + 2] == '_':
            p += 1
        if b[i + 2][j] == b[i][j + 2] and b[i + 2][j] == player and b[i + 1][j + 1] == '_':
            p += 1
        if b[i + 1][j + 1] == b[i][j + 2] and b[i + 1][j + 1] == player and b[i + 2][j] == '_':
            p += 1

        return p

    def evaluatePredifined(self, isMax):
        """
        This function implements the evaluation function for ultimate tic tac toe for predifined agent.
        input args:
        isMax(bool): boolean variable indicates whether it's maxPlayer or minPlayer.
                     True for maxPlayer, False for minPlayer
        output:
        score(float): estimated utility score for maxPlayer or minPlayer
        """
        # YOUR CODE HERE
        # Rule 1
        winner = self.checkWinner()
        if winner:
            if winner == 1 and isMax:
                return self.winnerMaxUtility
            if winner == -1 and not isMax:
                return self.winnerMinUtility

        # Rule 2
        score = 0.0

        if isMax:
            unblock = 500
            block = 100
            coner = 30
        else:
            unblock = 100
            block = 500
            coner = 30

        for start in self.globalIdx:
            score += self.checkBlocked(isMax, start) * block + self.checkUnblocked(isMax, start) * unblock

        # Rule 3
        if score == 0:
            if isMax:
                player = self.maxPlayer
            else:
                player = self.minPlayer

            b = self.board
            for start in self.globalIdx:
                i = start[0]
                j = start[1]
                corners = [b[i][j], b[i + 2][j], b[i][j + 2], b[i + 2][j + 2]]
                for c in corners:
                    if c == player:
                        score += coner

        if not isMax:
            score *= -1
        return score

    def evaluateDesigned(self, isMax):
        """
        This function implements the evaluation function for ultimate tic tac toe for your own agent.
        input args:
        isMax(bool): boolean variable indicates whether it's maxPlayer or minPlayer.
                     True for maxPlayer, False for minPlayer
        output:
        score(float): estimated utility score for maxPlayer or minPlayer
        """
        # YOUR CODE HERE
        # Rule 1
        winner = self.checkWinner()
        if winner:
            if winner == 1 and isMax:
                return self.winnerMaxUtility
            if winner == -1 and not isMax:
                return self.winnerMinUtility

        # Rule 2
        score = 0.0

        if isMax:
            unblock = 500
            block = 100
            coner = 80
            # oppo_unblock = -10
            # oppo_block = -50
            # oppo_coner = -3
        else:
            unblock = 100
            block = 500
            coner = 80
            # oppo_unblock = -10
            # oppo_block = -50
            # oppo_coner = -3
            edge = 60
            center = 100

        for start in self.globalIdx:
            score += self.checkBlocked(isMax, start) * block + self.checkUnblocked(isMax, start) * unblock
            # score += self.checkBlocked(not isMax, start)*oppo_block + self.checkUnblocked(not isMax, start)*oppo_unblock

        # Rule 3
        if score == 0:
            if isMax:
                player = self.maxPlayer
            else:
                player = self.minPlayer

            b = self.board
            for start in self.globalIdx:
                i = start[0]
                j = start[1]
                corners = [b[i][j], b[i + 2][j], b[i][j + 2], b[i + 2][j + 2]]
                for c in corners:
                    if c == player:
                        score += coner
                    # elif c != '_':
                    #     score += oppo_coner

                edges = [b[i + 1][j], b[i][j + 1], b[i + 2][j + 1], b[i + 2][j + 1]]
                for e in edges:
                    if e == player:
                        score += edge

                if b[i + 1][j + 1] == player:
                    score += center

        if not isMax:
            score *= -1
        return score

    def checkLocalEmptyspace(self, start):
        i = start[0]
        j = start[1]
        b = self.board

        for m in range(3):
            for n in range(3):
                if b[i + m][j + n] == '_':
                    return True

        return False

    def checkMovesLeft(self):
        """
        This function checks whether any legal move remains on the board.
        output:
        movesLeft(bool): boolean variable indicates whether any legal move remains
                        on the board.
        """
        # YOUR CODE HERE
        # local board
        start = self.globalIdx[self.startBoardIdx]
        if self.checkLocalEmptyspace(start):
            return (True, False)
        # other boards
        for local in self.globalIdx:
            if self.checkLocalEmptyspace(local):
                return (False, True)

        return (False, False)

    def checkWinner(self):
        # Return termimnal node status for maximizer player 1-win,0-tie,-1-lose
        """
        This function checks whether there is a winner on the board.
        output:
        winner(int): Return 0 if there is no winner.
                     Return 1 if maxPlayer is the winner.
                     Return -1 if miniPlayer is the winner.
        """
        # YOUR CODE HERE
        winner = 0

        b = self.board
        for start in self.globalIdx:
            i = start[0]
            j = start[1]

            # rows
            for row in range(3):
                if b[row + i][j] == b[row + i][j + 1] and b[row + i][j] == b[row + i][j + 2]:
                    if b[row + i][j] == self.maxPlayer:
                        winner = 1
                        return winner
                    elif b[row + i][j] == self.minPlayer:
                        winner = -1
                        return winner

            # columns
            for col in range(3):
                if b[i][col + j] == b[i + 1][col + j] and b[i][col + j] == b[i + 2][col + j]:
                    if b[i][col + j] == self.maxPlayer:
                        winner = 1
                        return winner
                    elif b[i][col + j] == self.minPlayer:
                        winner = -1
                        return winner

            # diagonals
            if b[i][j] == b[i + 1][j + 1] and b[i][j] == b[i + 2][j + 2]:
                if b[i + 1][j + 1] == self.maxPlayer:
                    winner = 1
                    return winner
                elif b[i + 1][j + 1] == self.minPlayer:
                    winner = -1
                    return winner

            if b[i + 2][j] == b[i + 1][j + 1] and b[i + 2][j] == b[i][j + 2]:
                if b[i + 1][j + 1] == self.maxPlayer:
                    winner = 1
                    return winner
                elif b[i + 1][j + 1] == self.minPlayer:
                    winner = -1
                    return winner

        return winner

    def alphabeta(self, depth, currBoardIdx, alpha, beta, isMax):
        """
        This function implements alpha-beta algorithm for ultimate tic-tac-toe game.
        input args:
        depth(int): current depth level
        currBoardIdx(int): current local board index
        alpha(float): alpha value
        beta(float): beta value
        isMax(bool):boolean variable indicates whether it's maxPlayer or minPlayer.
                     True for maxPlayer, False for minPlayer
        output:
        bestValue(float):the bestValue that current player may have
        """
        # YOUR CODE HERE
        bestValue = 0.0
        self.startBoardIdx = currBoardIdx
        start = self.globalIdx[self.startBoardIdx]
        checkMove = self.checkMovesLeft()
        if depth == 0 or checkMove == (False, False) or self.checkWinner():
            self.expandedNodes += 1
            return self.evaluatePredifined(self.currPlayer)
            # return self.evaluatePredifined(self.currPlayer)

        prune = False
        if isMax:
            bestValue = -inf
            if checkMove[0]:
                for i in range(3):
                    for j in range(3):
                        if self.board[start[0] + i][start[1] + j] == '_':
                            self.board[start[0] + i][start[1] + j] = self.maxPlayer
                            nextBoard = 3 * i + j
                            bestValue = max(bestValue, self.alphabeta(depth - 1, nextBoard, alpha, beta, not isMax))
                            alpha = max(alpha, bestValue)
                            if beta <= alpha:
                                prune = True
                            self.board[start[0] + i][start[1] + j] = '_'
                        if prune:
                            break
                    if prune:
                        break
                return bestValue
            elif checkMove[1]:
                for s in self.globalIdx:
                    if s == start:
                        continue

                    for i in range(3):
                        for j in range(3):
                            if self.board[s[0] + i][s[1] + j] == '_':
                                self.board[s[0] + i][s[1] + j] = self.maxPlayer
                                nextBoard = 3 * i + j
                                bestValue = max(bestValue, self.alphabeta(depth - 1, nextBoard, alpha, beta, not isMax))
                                alpha = max(alpha, bestValue)
                                if beta <= alpha:
                                    prune = True
                                self.board[s[0] + i][s[1] + j] = '_'
                            if prune:
                                break
                        if prune:
                            break
                return bestValue
        else:
            bestValue = inf
            if checkMove[0]:
                for i in range(3):
                    for j in range(3):
                        if self.board[start[0] + i][start[1] + j] == '_':
                            self.board[start[0] + i][start[1] + j] = self.minPlayer
                            nextBoard = 3 * i + j
                            bestValue = min(bestValue, self.alphabeta(depth - 1, nextBoard, alpha, beta, not isMax))
                            beta = min(beta, bestValue)
                            if beta <= alpha:
                                prune = True
                            self.board[start[0] + i][start[1] + j] = '_'
                        if prune:
                            break
                    if prune:
                        break
                return bestValue
            elif checkMove[1]:
                for s in self.globalIdx:
                    if s == start:
                        continue

                    for i in range(3):
                        for j in range(3):
                            if self.board[s[0] + i][s[1] + j] == '_':
                                self.board[s[0] + i][s[1] + j] = self.minPlayer
                                nextBoard = 3 * i + j
                                bestValue = min(bestValue, self.alphabeta(depth - 1, nextBoard, alpha, beta, not isMax))
                                beta = min(beta, bestValue)
                                if beta <= alpha:
                                    prune = True
                                self.board[s[0] + i][s[1] + j] = '_'
                            if prune:
                                break
                        if prune:
                            break
                return bestValue

        return bestValue

    def minimax(self, depth, currBoardIdx, isMax):
        """
        This function implements minimax algorithm for ultimate tic-tac-toe game.
        input args:
        depth(int): current depth level
        currBoardIdx(int): current local board index
        alpha(float): alpha value
        beta(float): beta value
        isMax(bool):boolean variable indicates whether it's maxPlayer or minPlayer.
                     True for maxPlayer, False for minPlayer
        output:
        bestValue(float):the bestValue that current player may have
        """
        # YOUR CODE HERE
        bestValue = 0.0
        self.startBoardIdx = currBoardIdx
        start = self.globalIdx[self.startBoardIdx]
        checkMove = self.checkMovesLeft()
        if depth == 0 or checkMove == (False, False) or self.checkWinner():
            self.expandedNodes += 1
            return self.evaluatePredifined(self.currPlayer)
            # return self.evaluateDesigned(self.currPlayer)

        if isMax:
            bestValue = -inf
            if checkMove[0]:
                for i in range(3):
                    for j in range(3):
                        if self.board[start[0] + i][start[1] + j] == '_':
                            self.board[start[0] + i][start[1] + j] = self.maxPlayer
                            nextBoard = 3 * i + j
                            bestValue = max(bestValue, self.minimax(depth - 1, nextBoard, False))
                            self.board[start[0] + i][start[1] + j] = '_'
                return bestValue
            elif checkMove[1]:
                for s in self.globalIdx:
                    if s == start:
                        continue

                    for i in range(3):
                        for j in range(3):
                            if self.board[s[0] + i][s[1] + j] == '_':
                                self.board[s[0] + i][s[1] + j] = self.maxPlayer
                                nextBoard = 3 * i + j
                                bestValue = max(bestValue, self.minimax(depth - 1, nextBoard, False))
                                self.board[s[0] + i][s[1] + j] = '_'
                return bestValue
        else:
            bestValue = inf
            if checkMove[0]:
                for i in range(3):
                    for j in range(3):
                        if self.board[start[0] + i][start[1] + j] == '_':
                            self.board[start[0] + i][start[1] + j] = self.minPlayer
                            nextBoard = 3 * i + j
                            bestValue = min(bestValue, self.minimax(depth - 1, nextBoard, True))
                            self.board[start[0] + i][start[1] + j] = '_'
                return bestValue
            elif checkMove[1]:
                for s in self.globalIdx:
                    if s == start:
                        continue

                    for i in range(3):
                        for j in range(3):
                            if self.board[s[0] + i][s[1] + j] == '_':
                                self.board[s[0] + i][s[1] + j] = self.minPlayer
                                nextBoard = 3 * i + j
                                bestValue = min(bestValue, self.minimax(depth - 1, nextBoard, True))
                                self.board[s[0] + i][s[1] + j] = '_'
                return bestValue

        return bestValue

    def findMove(self, isMax, checkMove, isMinimax):
        move = (-1, -1)
        value = 0

        if isMax:
            player = self.maxPlayer
            bound = -inf
        else:
            player = self.minPlayer
            bound = inf

        currBoard = self.startBoardIdx
        depthIdx = self.maxDepth - 1
        start = self.globalIdx[currBoard]
        if checkMove[0]:
            for i in range(3):
                for j in range(3):
                    if self.board[start[0] + i][start[1] + j] == '_':
                        self.board[start[0] + i][start[1] + j] = player
                        nextBoard = 3 * i + j
                        if isMinimax:
                            value = self.minimax(depthIdx, nextBoard, not isMax)
                        else:
                            value = self.alphabeta(depthIdx, nextBoard, -inf, inf, not isMax)
                        if isMax and value > bound:
                            move = (start[0] + i, start[1] + j)
                            bound = value
                        elif (not isMax) and value < bound:
                            move = (start[0] + i, start[1] + j)
                            bound = value
                        self.board[start[0] + i][start[1] + j] = '_'
        elif checkMove[1]:
            for s in self.globalIdx:
                if s == start:
                    continue

                for i in range(3):
                    for j in range(3):
                        if self.board[s[0] + i][s[1] + j] == '_':
                            self.board[s[0] + i][s[1] + j] = player
                            nextBoard = 3 * i + j
                            if isMinimax:
                                value = self.minimax(depthIdx, nextBoard, not isMax)
                            else:
                                value = self.alphabeta(depthIdx, nextBoard, -inf, inf, not isMax)
                            if isMax and value > bound:
                                move = (s[0] + i, s[1] + j)
                                bound = value
                            elif (not isMax) and value < bound:
                                move = (s[0] + i, s[0] + j)
                                bound = value
                            self.board[s[0] + i][s[1] + j] = '_'
        return move, bound

    ###-------------------------------------------------------------------------------------------------------------
    def alphabetaDesigned(self, depth, currBoardIdx, alpha, beta, isMax):
        """
        This function implements alpha-beta algorithm for ultimate tic-tac-toe game.
        input args:
        depth(int): current depth level
        currBoardIdx(int): current local board index
        alpha(float): alpha value
        beta(float): beta value
        isMax(bool):boolean variable indicates whether it's maxPlayer or minPlayer.
                     True for maxPlayer, False for minPlayer
        output:
        bestValue(float):the bestValue that current player may have
        """
        # YOUR CODE HERE
        bestValue = 0.0
        self.startBoardIdx = currBoardIdx
        start = self.globalIdx[self.startBoardIdx]
        checkMove = self.checkMovesLeft()
        if depth == 0 or checkMove == (False, False) or self.checkWinner():
            self.expandedNodes += 1
            # return self.evaluatePredifined(self.currPlayer)
            return self.evaluateDesigned(self.currPlayer)

        prune = False
        if isMax:
            bestValue = -inf
            if checkMove[0]:
                for i in range(3):
                    for j in range(3):
                        if self.board[start[0] + i][start[1] + j] == '_':
                            self.board[start[0] + i][start[1] + j] = self.maxPlayer
                            nextBoard = 3 * i + j
                            bestValue = max(bestValue,
                                            self.alphabetaDesigned(depth - 1, nextBoard, alpha, beta, not isMax))
                            alpha = max(alpha, bestValue)
                            if beta <= alpha:
                                prune = True
                            self.board[start[0] + i][start[1] + j] = '_'
                        if prune:
                            break
                    if prune:
                        break
                return bestValue
            elif checkMove[1]:
                for s in self.globalIdx:
                    if s == start:
                        continue

                    for i in range(3):
                        for j in range(3):
                            if self.board[s[0] + i][s[1] + j] == '_':
                                self.board[s[0] + i][s[1] + j] = self.maxPlayer
                                nextBoard = 3 * i + j
                                bestValue = max(bestValue,
                                                self.alphabetaDesigned(depth - 1, nextBoard, alpha, beta, not isMax))
                                alpha = max(alpha, bestValue)
                                if beta <= alpha:
                                    prune = True
                                self.board[s[0] + i][s[1] + j] = '_'
                            if prune:
                                break
                        if prune:
                            break
                return bestValue
        else:
            bestValue = inf
            if checkMove[0]:
                for i in range(3):
                    for j in range(3):
                        if self.board[start[0] + i][start[1] + j] == '_':
                            self.board[start[0] + i][start[1] + j] = self.minPlayer
                            nextBoard = 3 * i + j
                            bestValue = min(bestValue,
                                            self.alphabetaDesigned(depth - 1, nextBoard, alpha, beta, not isMax))
                            beta = min(beta, bestValue)
                            if beta <= alpha:
                                prune = True
                            self.board[start[0] + i][start[1] + j] = '_'
                        if prune:
                            break
                    if prune:
                        break
                return bestValue
            elif checkMove[1]:
                for s in self.globalIdx:
                    if s == start:
                        continue

                    for i in range(3):
                        for j in range(3):
                            if self.board[s[0] + i][s[1] + j] == '_':
                                self.board[s[0] + i][s[1] + j] = self.minPlayer
                                nextBoard = 3 * i + j
                                bestValue = min(bestValue,
                                                self.alphabetaDesigned(depth - 1, nextBoard, alpha, beta, not isMax))
                                beta = min(beta, bestValue)
                                if beta <= alpha:
                                    prune = True
                                self.board[s[0] + i][s[1] + j] = '_'
                            if prune:
                                break
                        if prune:
                            break
                return bestValue

        return bestValue

    def minimaxDesigned(self, depth, currBoardIdx, isMax):
        """
        This function implements minimax algorithm for ultimate tic-tac-toe game.
        input args:
        depth(int): current depth level
        currBoardIdx(int): current local board index
        alpha(float): alpha value
        beta(float): beta value
        isMax(bool):boolean variable indicates whether it's maxPlayer or minPlayer.
                     True for maxPlayer, False for minPlayer
        output:
        bestValue(float):the bestValue that current player may have
        """
        # YOUR CODE HERE
        bestValue = 0.0
        self.startBoardIdx = currBoardIdx
        start = self.globalIdx[self.startBoardIdx]
        checkMove = self.checkMovesLeft()
        if depth == 0 or checkMove == (False, False) or self.checkWinner():
            self.expandedNodes += 1
            # return self.evaluatePredifined(self.currPlayer)
            return self.evaluateDesigned(self.currPlayer)

        if isMax:
            bestValue = -inf
            if checkMove[0]:
                for i in range(3):
                    for j in range(3):
                        if self.board[start[0] + i][start[1] + j] == '_':
                            self.board[start[0] + i][start[1] + j] = self.maxPlayer
                            nextBoard = 3 * i + j
                            bestValue = max(bestValue, self.minimaxDesigned(depth - 1, nextBoard, False))
                            self.board[start[0] + i][start[1] + j] = '_'
                return bestValue
            elif checkMove[1]:
                for s in self.globalIdx:
                    if s == start:
                        continue

                    for i in range(3):
                        for j in range(3):
                            if self.board[s[0] + i][s[1] + j] == '_':
                                self.board[s[0] + i][s[1] + j] = self.maxPlayer
                                nextBoard = 3 * i + j
                                bestValue = max(bestValue, self.minimaxDesigned(depth - 1, nextBoard, False))
                                self.board[s[0] + i][s[1] + j] = '_'
                return bestValue
        else:
            bestValue = inf
            if checkMove[0]:
                for i in range(3):
                    for j in range(3):
                        if self.board[start[0] + i][start[1] + j] == '_':
                            self.board[start[0] + i][start[1] + j] = self.minPlayer
                            nextBoard = 3 * i + j
                            bestValue = min(bestValue, self.minimaxDesigned(depth - 1, nextBoard, True))
                            self.board[start[0] + i][start[1] + j] = '_'
                return bestValue
            elif checkMove[1]:
                for s in self.globalIdx:
                    if s == start:
                        continue

                    for i in range(3):
                        for j in range(3):
                            if self.board[s[0] + i][s[1] + j] == '_':
                                self.board[s[0] + i][s[1] + j] = self.minPlayer
                                nextBoard = 3 * i + j
                                bestValue = min(bestValue, self.minimaxDesigned(depth - 1, nextBoard, True))
                                self.board[s[0] + i][s[1] + j] = '_'
                return bestValue

        return bestValue

    def findMoveDesigned(self, isMax, checkMove, isMinimax):
        move = (-1, -1)
        value = 0

        if isMax:
            player = self.maxPlayer
            bound = -inf
        else:
            player = self.minPlayer
            bound = inf

        currBoard = self.startBoardIdx
        depthIdx = self.maxDepth - 1
        start = self.globalIdx[currBoard]
        if checkMove[0]:
            for i in range(3):
                for j in range(3):
                    if self.board[start[0] + i][start[1] + j] == '_':
                        self.board[start[0] + i][start[1] + j] = player
                        nextBoard = 3 * i + j
                        if isMinimax:
                            value = self.minimaxDesigned(depthIdx, nextBoard, not isMax)
                        else:
                            value = self.alphabetaDesigned(depthIdx, nextBoard, -inf, inf, not isMax)
                        if isMax and value > bound:
                            move = (start[0] + i, start[1] + j)
                            bound = value
                        elif (not isMax) and value < bound:
                            move = (start[0] + i, start[1] + j)
                            bound = value
                        self.board[start[0] + i][start[1] + j] = '_'
        elif checkMove[1]:
            for s in self.globalIdx:
                if s == start:
                    continue

                for i in range(3):
                    for j in range(3):
                        if self.board[s[0] + i][s[1] + j] == '_':
                            self.board[s[0] + i][s[1] + j] = player
                            nextBoard = 3 * i + j
                            if isMinimax:
                                value = self.minimaxDesigned(depthIdx, nextBoard, not isMax)
                            else:
                                value = self.alphabetaDesigned(depthIdx, nextBoard, -inf, inf, not isMax)
                            if isMax and value > bound:
                                move = (s[0] + i, s[1] + j)
                                bound = value
                            elif (not isMax) and value < bound:
                                move = (s[0] + i, s[0] + j)
                                bound = value
                            self.board[s[0] + i][s[1] + j] = '_'
        return move, bound

    ###-------------------------------------------------------------------------------------------------------------------------------------

    def playGamePredifinedAgent(self, maxFirst, isMinimaxOffensive, isMinimaxDefensive):
        """
        This function implements the processes of the game of predifined offensive agent vs defensive agent.
        input args:
        maxFirst(bool): boolean variable indicates whether maxPlayer or minPlayer plays first.
                        True for maxPlayer plays first, and False for minPlayer plays first.
        isMinimaxOffensive(bool):boolean variable indicates whether it's using minimax or alpha-beta pruning algorithm for offensive agent.
                        True is minimax and False is alpha-beta.
        isMinimaxDefensive(bool):boolean variable indicates whether it's using minimax or alpha-beta pruning algorithm for defensive agent.
                        True is minimax and False is alpha-beta.
        output:
        bestMove(list of tuple): list of bestMove coordinates at each step
        bestValue(list of float): list of bestValue at each move
        expandedNodes(list of int): list of expanded nodes at each move
        gameBoards(list of 2d lists): list of game board positions at each move
        winner(int): 1 for maxPlayer is the winner, -1 for minPlayer is the winner, and 0 for tie.
        """
        # YOUR CODE HERE
        self.board = [['_', '_', '_', '_', '_', '_', '_', '_', '_'],
                      ['_', '_', '_', '_', '_', '_', '_', '_', '_'],
                      ['_', '_', '_', '_', '_', '_', '_', '_', '_'],
                      ['_', '_', '_', '_', '_', '_', '_', '_', '_'],
                      ['_', '_', '_', '_', '_', '_', '_', '_', '_'],
                      ['_', '_', '_', '_', '_', '_', '_', '_', '_'],
                      ['_', '_', '_', '_', '_', '_', '_', '_', '_'],
                      ['_', '_', '_', '_', '_', '_', '_', '_', '_'],
                      ['_', '_', '_', '_', '_', '_', '_', '_', '_']]
        bestMove = []
        bestValue = []
        gameBoards = []
        expandedNodes = []
        self.startBoardIdx = 4
        winner = 0

        while (True):
            # self.expadedNodes = 0
            winner = self.checkWinner()
            if winner:
                print("Winner Detected!")
                if winner == 1:
                    print("MaxPlayer Wins!")
                else:
                    print("MinPlayer Wins!")
                break

            checkMove = self.checkMovesLeft()
            if not (checkMove[0] or checkMove[1]):
                print("Draw Detected")
                break
            # self.currPlayer: Mark the current player to identify evaluation criteria.
            if maxFirst:
                # print("X's turn to play!")
                self.currPlayer = True
                bestmove, score = self.findMove(maxFirst, checkMove, isMinimaxOffensive)
                self.board[bestmove[0]][bestmove[1]] = self.maxPlayer
            else:
                # print("O's turn to play!")
                self.currPlayer = False
                bestmove, score = self.findMove(maxFirst, checkMove, isMinimaxDefensive)
                self.board[bestmove[0]][bestmove[1]] = self.minPlayer

            # self.printGameBoard()
            bestValue.append(score)
            gameBoards.append(self.board)
            bestMove.append(bestmove)
            expandedNodes.append(self.expandedNodes)
            maxFirst = not maxFirst

            self.startBoardIdx = 3 * (bestmove[0] % 3) + bestmove[1] % 3
        self.printGameBoard()
        return gameBoards, bestMove, expandedNodes, bestValue, winner

    def playGameYourAgent(self, isMinimaxOffensive, isMinimaxDefensive):
        """
        This function implements the processes of the game of your own agent vs predifined offensive agent.
        input args:
        output:
        bestMove(list of tuple): list of bestMove coordinates at each step
        gameBoards(list of 2d lists): list of game board positions at each move
        winner(int): 1 for maxPlayer is the winner, -1 for minPlayer is the winner, and 0 for tie.
        """
        # YOUR CODE HERE
        self.board = [['_', '_', '_', '_', '_', '_', '_', '_', '_'],
                      ['_', '_', '_', '_', '_', '_', '_', '_', '_'],
                      ['_', '_', '_', '_', '_', '_', '_', '_', '_'],
                      ['_', '_', '_', '_', '_', '_', '_', '_', '_'],
                      ['_', '_', '_', '_', '_', '_', '_', '_', '_'],
                      ['_', '_', '_', '_', '_', '_', '_', '_', '_'],
                      ['_', '_', '_', '_', '_', '_', '_', '_', '_'],
                      ['_', '_', '_', '_', '_', '_', '_', '_', '_'],
                      ['_', '_', '_', '_', '_', '_', '_', '_', '_']]
        bestMove = []
        gameBoards = []
        winner = 0
        self.startBoardIdx = randint(0, 8)
        maxFirst = choice([True, False])
        print("First Player Is:", maxFirst)
        print("Starting Board Is:", self.startBoardIdx)

        while (True):
            # self.expadedNodes = 0
            winner = self.checkWinner()
            if winner:
                print("Winner Detected!")
                if winner == 1:
                    print("MaxPlayer Wins!")
                else:
                    print("MinPlayer Wins!")
                break

            checkMove = self.checkMovesLeft()
            if not (checkMove[0] or checkMove[1]):
                print("Draw Detected")
                break
            # self.currPlayer: Mark the current player to identify evaluation criteria.
            if maxFirst:
                # print("X's turn to play!")
                self.currPlayer = True
                bestmove = self.findMove(maxFirst, checkMove, isMinimaxOffensive)[0]
                self.board[bestmove[0]][bestmove[1]] = self.maxPlayer
            else:
                # print("O's turn to play!")
                self.currPlayer = False
                bestmove = self.findMoveDesigned(maxFirst, checkMove, isMinimaxDefensive)[0]
                self.board[bestmove[0]][bestmove[1]] = self.minPlayer

            # self.printGameBoard()
            gameBoards.append(self.board)
            bestMove.append(bestmove)
            maxFirst = not maxFirst

            self.startBoardIdx = 3 * (bestmove[0] % 3) + bestmove[1] % 3
        return gameBoards, bestMove, winner

    def playGameHuman(self, isMinimaxDefensive):
        """
        This function implements the processes of the game of your own agent vs a human.
        output:
        bestMove(list of tuple): list of bestMove coordinates at each step
        gameBoards(list of 2d lists): list of game board positions at each move
        winner(int): 1 for maxPlayer is the winner, -1 for minPlayer is the winner, and 0 for tie.
        """
        # YOUR CODE HERE
        self.board = [['_', '_', '_', '_', '_', '_', '_', '_', '_'],
                      ['_', '_', '_', '_', '_', '_', '_', '_', '_'],
                      ['_', '_', '_', '_', '_', '_', '_', '_', '_'],
                      ['_', '_', '_', '_', '_', '_', '_', '_', '_'],
                      ['_', '_', '_', '_', '_', '_', '_', '_', '_'],
                      ['_', '_', '_', '_', '_', '_', '_', '_', '_'],
                      ['_', '_', '_', '_', '_', '_', '_', '_', '_'],
                      ['_', '_', '_', '_', '_', '_', '_', '_', '_'],
                      ['_', '_', '_', '_', '_', '_', '_', '_', '_']]
        bestMove = []
        gameBoards = []
        winner = 0
        self.startBoardIdx = randint(0, 8)
        maxFirst = choice([True, False])
        print("First Player Is:", maxFirst)
        print("Starting Board Is:", self.startBoardIdx)

        while (True):
            # self.expadedNodes = 0
            winner = self.checkWinner()
            print(winner)
            if winner:
                break

            checkMove = self.checkMovesLeft()
            if not (checkMove[0] or checkMove[1]):
                # print("Draw Detected")
                break
            # self.currPlayer: Mark the current player to identify evaluation criteria.
            if maxFirst:
                print("X's turn to play!")
                self.currPlayer = True
                bestmove = self.human()
                self.board[bestmove[0]][bestmove[1]] = self.maxPlayer
            else:
                print("O's turn to play!")
                self.currPlayer = False
                bestmove = self.findMoveDesigned(maxFirst, checkMove, isMinimaxDefensive)[0]
                self.board[bestmove[0]][bestmove[1]] = self.minPlayer

            self.printGameBoard()
            gameBoards.append(self.board)
            bestMove.append(bestmove)
            maxFirst = not maxFirst

            self.startBoardIdx = 3 * (bestmove[0] % 3) + bestmove[1] % 3

        return gameBoards, bestMove, winner

    def human(self):
        print('You need to play in local board:', self.startBoardIdx)
        i = int(input('rowIdx(0/1/2)? Answer:'))
        j = int(input('colIdx(0/1/2)? Answer:'))
        print("")
        start = self.globalIdx[self.startBoardIdx]
        if (i < 0 or i > 2) or (j < 0 or j > 2):
            print("Invalid Index! I'll beat you up!")
            return self.human()
        if self.board[start[0] + i][start[1] + j] != "_":
            print("Already taken! I'll beat you up!")
            return self.human()
        return start[0] + i, start[1] + j


def draw_board(board):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 9)

    # 绘制棋盘的线
    for x in range(1, 9):
        if x % 3 == 0:
            ax.axvline(x=x, color='black', linewidth=4)
            ax.axhline(y=x, color='black', linewidth=4)
        else:
            ax.axvline(x=x, color='gray', linewidth=1)
            ax.axhline(y=x, color='gray', linewidth=1)

    for i in range(9):
        for j in range(9):
            # 标注方块颜色
            if board[i][j] == 'X':
                ax.add_patch(plt.Rectangle((j, 8 - i), 1, 1, facecolor='lightgreen'))
            elif board[i][j] == 'O':
                ax.add_patch(plt.Rectangle((j, 8 - i), 1, 1, facecolor='lightblue'))

            # 绘制 'X' 和 'O'
            if board[i][j] == 'X':
                ax.text(j + 0.5, 8.5 - i, 'X', ha='center', va='center', fontsize=20)
            elif board[i][j] == 'O':
                ax.text(j + 0.5, 8.5 - i, 'O', ha='center', va='center', fontsize=20)

    ax.set_xticks([])
    ax.set_yticks([])
    plt.gca().invert_yaxis()  # 将坐标系翻转，以便左上角为(0,0)
    plt.show()


if __name__ == "__main__":
    uttt = ultimateTicTacToe()
    # feel free to write your own test code
    # gameBoards, bestMove, expandedNodes, bestValue, winner=uttt.playGamePredifinedAgent(True,True,True)
    # print(expandedNodes)
    # # draw(uttt.board)
    # gameBoards, bestMove, expandedNodes, bestValue, winner=uttt.playGamePredifinedAgent(True,False,False)
    # print(expandedNodes)
    # draw(uttt.board)
    # gameBoards, bestMove, expandedNodes, bestValue, winner=uttt.playGamePredifinedAgent(False,True,False)
    # print(expandedNodes)
    # draw(uttt.board)
    # gameBoards, bestMove, expandedNodes, bestValue, winner=uttt.playGamePredifinedAgent(False,False,True)
    # print(expandedNodes)
    # draw(uttt.board)

    # gameBoards, bestMove, winner=uttt.playGameYourAgent(True, True)

    gameboards, bestMove, winner = uttt.playGameHuman(False)
    draw_board(uttt.board)  # draw the final board

    # wins=0
    # times=100

    # for i in range(times):
    #     gameBoards, bestMove, winner = uttt.playGameYourAgent(False, False)
    #     print("Winner is:", winner)
    #     print("------")
    #     if winner == -1:
    #         wins += 1

    # print(wins/times*100) 

    # if winner == 1:
    #     print("The winner is maxPlayer!!!")
    # elif winner == -1:
    #     print("The winner is minPlayer!!!")
    # else:
    #     print("Tie. No winner:(")
