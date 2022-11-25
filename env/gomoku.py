import numpy as np
import os

EMPTY = "[ ]"
PLAYER1 = "[O]"
PLAYER2 = "[X]"
debug = False
MAX_ROUNDS = 2


# returns the initial state of the game, with all the board empty
def get_initial_state():
    state = []
    for i in range(15):
        state.append([EMPTY for j in range(15)])
    return state


# given a position (pos) [x,y] of the board, returns if its empty
def is_position_available(state, pos):
    return state[pos[0]][pos[1]] == EMPTY


def get_positions_bounded(state, moves):
    positions = []
    for move in moves:
        move_x = move[0]
        move_y = move[1]
        for i in range(5):
            for j in range(5):
                new_bound_x = move_x + (i - 2)
                new_bound_y = move_y + (j - 2)
                if (new_bound_x >= 0) and (new_bound_y >= 0) and (new_bound_x < 15) and (new_bound_y < 15):
                    if ([new_bound_x, new_bound_y] not in positions) and (state[new_bound_x][new_bound_y] == EMPTY):
                        positions.append([new_bound_x, new_bound_y])
    return positions


def get_vertical_from_position(state, pos):
    a = np.array(state)
    return a[:, pos[1]]


def get_horizontal_from_position(state, pos):
    a = np.array(state)
    return a[pos[0], :]


def get_sequences_from_positions(state, moves):
    sequences = []
    visited = []
    for move in moves:
        if move[1] not in visited:
            vertical = get_vertical_from_position(state, move)
            sequence = get_sequences_in_array(vertical)
            visited.append(move[1])
            if len(sequence) > 0:
                sequences.extend(sequence)

        if move[0] + 15 not in visited:
            horizontal = get_horizontal_from_position(state, move)
            sequence = get_sequences_in_array(horizontal)
            visited.append(move[0] + 15)
            if len(sequence) > 0:
                sequences.extend(sequence)

    return sequences


# returns sequences in an array
def get_sequences_in_array(array):
    sequences = []
    tmp_seq = []
    tmp_opening = 0
    for i, item in enumerate(array):
        if i > 0:
            last_item = array[i - 1]
            if item != EMPTY:
                if last_item != item:
                    if last_item == EMPTY:
                        tmp_opening = 1
                        tmp_seq.append(item)
                    else:
                        if (len(tmp_seq) > 1):
                            sequences.append([tmp_seq[0], tmp_opening, len(tmp_seq)])
                        tmp_seq = []
                        tmp_opening = 0
                else:
                    if (len(tmp_seq) < 1):
                        tmp_seq.append(last_item)
                    tmp_seq.append(item)
            elif last_item != item:
                if len(tmp_seq) > 1:
                    sequences.append([tmp_seq[0], tmp_opening + 1, len(tmp_seq)])
                tmp_seq = []
                tmp_opening = 0
    if len(tmp_seq) > 1:
        sequences.append([tmp_seq[0], tmp_opening, len(tmp_seq)])
    return sequences


# returns an array of sequences
def get_diagonal_sequences(state):
    sequences = []
    x, y = 15, 15
    a = np.array(state)
    diags = [a[::-1, :].diagonal(i) for i in range(-a.shape[0] + 1, a.shape[1])]
    diags.extend(a.diagonal(i) for i in range(a.shape[1] - 1, -a.shape[0], -1))
    diagonals = [n.tolist() for n in diags]
    for diagonal in diagonals:
        sequences += get_sequences_in_array(diagonal)
    return sequences


# returns score given the length of a sequence
def get_sequence_score(length):
    if length == 2:
        return 1
    elif length == 3:
        return 812
    elif length == 4:
        return 591136
    elif length == 5:
        return 383056128


def get_all_sequences(state, moves):
    all_sequences = (get_sequences_from_positions(state, moves))
    for sequence in get_diagonal_sequences(state):
        all_sequences.append(sequence)
    return all_sequences


def game_over(winner, pvp):
    print("GAME OVER")
    print("\nPLAYER " + str(winner) + " WINS" if winner != EMPTY else "No more available positions. It's a tie!")

    exit = False
    start_game(pvp)


# user = PLAYER1 or PLAYER2
def make_move(state, pos, user):
    if is_position_available(state, pos):
        state[pos[0]][pos[1]] = user
        return True
    return False


def unmake_move(state, move):
    state[move[0]][move[1]] = EMPTY


def exists_winner(state, moves):
    all_sequences = get_all_sequences(state, moves)
    for sequence in all_sequences:
        if len(sequence) > 0:
            if sequence[2] == 5:
                return sequence[0]
    return EMPTY


def print_header():
    os.system('tput reset')
    print(" === === === === === ===  PYMOKU  === === === === === === ===")


# prints a state of the board
def print_state(state):
    s = " "
    for j in range(15):
        s += "  " + str(j).zfill(2)
    print(s)
    i = 0;
    for row in state:
        string = ""
        for column in row:
            string += column + " "
        print(str(i).zfill(2) + " " + string)
        i += 1


def input_position(turn, state):
    while True:
        try:
            print("Player " + turn + "'s turn:")
            row = int(input("row: "))
            col = int(input("col: "))
            if (row > 15 or col > 15):
                print("Please insert values between 0 and 14")
            elif not is_position_available(state, [row, col]):
                print("Position is busy")
            else:
                return [row, col]
        except IndexError:
            print("Please insert values between 0 and 14")
        except ValueError:
            print("Please insert values between 0 and 14")
    return []


def start_game(pvp):
    round_number = 1
    moves = []
    state = get_initial_state()
    turn = PLAYER2
    winner = EMPTY
    print_header()
    print_state(state)
    while winner == EMPTY:
        if len(moves) == 225:
            break
        if (turn == PLAYER2) and (len(moves) == 0):
            make_move(state, [7, 7], turn)
        else:
            move = input_position(turn, state)
            make_move(state, move, turn)
            moves.append(move)
        round_number += 1
        winner = exists_winner(state, moves)
        turn = PLAYER2 if turn == PLAYER1 else PLAYER1
        print_header()
        print_state(state)
    game_over(winner, pvp)


class GomokuGame:
    def __init__(self, initial_state, dbg=False):
        self.round_number = 1
        self.moves = []
        self.state = initial_state
        self.winner = EMPTY
        self.turn = PLAYER2

        if dbg:
            print("starting game")
            print_state(self.state)

    def get_current_turn(self):
        return self.turn

    def get_current_move_count(self):
        return self.moves

    def get_winner(self):
        exists_winner(self.state, self.moves)

    def print_board(self):
        print_state(self.state)

    # player is PLAYER1/PLAYER2
    def make_move(self, move, player):
        if move[0] > 15 or move[1] > 15:
            print("Please insert values between 0 and 14")
        elif not is_position_available(self.state, move):
            print("Position is busy")
        else:
            make_move(self.state, move, player)
            self.moves.append(move)
            self.round_number += 1
            self.winner = exists_winner(self.state, self.moves)
            self.turn = PLAYER2 if self.turn == PLAYER1 else PLAYER1