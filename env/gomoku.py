from env.utils import *


class GomokuGame:
    def __init__(self, init_state):
        self.round_number = 1
        self.moves = []
        self.state = init_state

        self.winner = EMPTY
        # pick first player with 50% prob
        self.bad_tries = 0

    def get_current_turn(self):
        return self.turn

    def get_current_move_count(self):
        return len(self.moves)

    def get_winner(self):
        return exists_winner(self.state, self.moves)

    def get_board_state(self):
        return np.array(self.state)

    def print_game_state(self):
        """board state with steps"""

        print(f"Number of bad attempts: {self.bad_tries}")
        visual_state = np.zeros([BOARD_SIZE, BOARD_SIZE])
        index = 1
        for (x, y) in self.moves:
            # opponent gets the positive index
            visual_state[x, y] = index if (index % OPPONENT) != 0 else -index
            index += 1

        np.set_printoptions(formatter={'float': '{: 3.0f}'.format})
        print(visual_state)

    # players are PLAYER/OPPONENT,
    def make_move(self, move, player):
        if move[0] > BOARD_SIZE or move[1] > BOARD_SIZE:
            # does not happen irl
            print("Please insert values between 0 and 14")
            return np.array(self.state).flatten(), -BOARD_SIZE*BOARD_SIZE, True
        elif not is_position_available(self.state, move):
            self.bad_tries += 1
            return np.array(self.state).flatten(), -100.00 / BOARD_SIZE*BOARD_SIZE, False
        else:
            make_move(self.state, move, player)
            self.moves.append(move)
            self.round_number += 1
            self.winner = exists_winner(self.state, self.moves)

            state_array = np.array(self.state)
            state_array = state_array.flatten()

            if self.winner is EMPTY:
                # return over if step count == num
                return state_array, -2.0000 / BOARD_SIZE*BOARD_SIZE, \
                       (self.get_current_move_count() == (BOARD_SIZE * BOARD_SIZE))
            else:
                # print("__________________WIN______________________")
                print(f"The winner is: {self.turn}")
                print(f"The step count is: {self.get_current_move_count()}")
                self.bad_tries = 0
                return state_array, 5, True
        # todo: draw situation?


# Test game
if __name__ == '__main__':
    initial_state = get_initial_state()
    game = GomokuGame(initial_state)
    game.make_move([7, 7], OPPONENT)
    game.make_move([1, 8], PLAYER)
    game.make_move([7, 8], OPPONENT)
    print(f"winner = {game.get_winner()}")

    game.make_move([7, 9], OPPONENT)
    game.make_move([2, 8], PLAYER)
    game.make_move([7, 10], OPPONENT)
    game.make_move([2, 9], PLAYER)
    game.make_move([7, 11], OPPONENT)
    print(game.make_move([1, 9], PLAYER))

    print(f"winner = {game.get_winner()}")

    game.make_move([3, 11], PLAYER)

    # game.print_board()
