import copy
import math
import random

from exceptions import AgentException


class RandomAgent:
    def __init__(self, my_token='o'):
        self.my_token = my_token

    def decide(self, connect4):
        if connect4.who_moves != self.my_token:
            raise AgentException('not my round')
        score_move_list = self.minmax(connect4, 6, True, -math.inf, math.inf)
        return score_move_list[1]

    def minmax(self, connect4, depth, maximazing, alfa, beta):
        if self.my_token == 'o':
            enemy = 'x'
        else:
            enemy = 'o'
        if connect4.check_game_over():
            if connect4.wins == self.my_token:
                return [10000, None]
            elif connect4.wins == enemy:
                return [-10000, None]
            else:
                return [0, None]
        elif depth == 0:
            score = 0
            score += 3 * connect4.center_column().count(self.my_token)
            scores = connect4.iter_fours()
            for x in scores:
                a = x.count(self.my_token)
                b = x.count(enemy)
                if a == 3:
                    score += 5
                if a == 2:
                    score += 2
                if b == 3:
                    score -= 4
            return [score, None]
        if maximazing:
            best_score = -math.inf
            best_move = None
            moves = connect4.possible_drops()
            for move in moves:
                board_copy = copy.deepcopy(connect4)
                board_copy.drop_token(move)
                move_list = self.minmax(board_copy, depth - 1, False, alfa, beta)
                if move_list[0] > best_score:
                    best_score = move_list[0]
                    best_move = move
                alfa = max(alfa, best_score)
                if alfa >= beta:
                    break
            return [best_score, best_move]
        else:
            best_score = math.inf
            best_move = None
            moves = connect4.possible_drops()
            for move in moves:
                board_copy = copy.deepcopy(connect4)
                board_copy.drop_token(move)
                score_move_list = self.minmax(board_copy, depth - 1, True, alfa, beta)
                if score_move_list[0] < best_score:
                    best_score = score_move_list[0]
                    best_move = move
                beta = min(beta, best_score)
                if alfa >= beta:
                    break
            return [best_score, best_move]