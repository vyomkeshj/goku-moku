import numpy as np

# EMPTY = "[ ]"
# PLAYER1 = "[O]"
# PLAYER2 = "[X]"

BOARD_SIZE = 3
PATTERN_SIZE = 3

EMPTY = 0
PLAYER1 = 1
PLAYER2 = 2


# returns the initial state of the game, with all the board empty
def get_initial_state():
    state = []
    for i in range(BOARD_SIZE):
        state.append([EMPTY for j in range(BOARD_SIZE)])
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
                if (new_bound_x >= 0) and (new_bound_y >= 0) and (new_bound_x < BOARD_SIZE) and (new_bound_y < BOARD_SIZE):
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

        if move[0] + BOARD_SIZE not in visited:
            horizontal = get_horizontal_from_position(state, move)
            sequence = get_sequences_in_array(horizontal)
            visited.append(move[0] + BOARD_SIZE)
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
    x, y = BOARD_SIZE, BOARD_SIZE
    a = np.array(state)
    diags = [a[::-1, :].diagonal(i) for i in range(-a.shape[0] + 1, a.shape[1])]
    diags.extend(a.diagonal(i) for i in range(a.shape[1] - 1, -a.shape[0], -1))
    diagonals = [n.tolist() for n in diags]
    for diagonal in diagonals:
        sequences += get_sequences_in_array(diagonal)
    return sequences


def get_all_sequences(state, moves):
    all_sequences = (get_sequences_from_positions(state, moves))
    for sequence in get_diagonal_sequences(state):
        all_sequences.append(sequence)
    return all_sequences


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
            if sequence[2] == PATTERN_SIZE:
                return sequence[0]
    return EMPTY


# prints a state of the board
def print_state(state):
    print(np.array(state))


def input_position(turn, state):
    while True:
        try:
            print("Player " + turn + "'s turn:")
            row = int(input("row: "))
            col = int(input("col: "))
            if row > BOARD_SIZE or col > BOARD_SIZE:
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
