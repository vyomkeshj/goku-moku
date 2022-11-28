from env.utils import *


class GomokuGame:
    def __init__(self, init_state):
        self.round_number = 1
        self.moves = []
        self.state = init_state
        self.winner = EMPTY
        self.turn = PLAYER1
        self.bad_tries = 0

    def get_current_turn(self):
        return self.turn

    def get_current_move_count(self):
        return self.moves

    def get_winner(self):
        return exists_winner(self.state, self.moves)

    def get_board_state(self):
        return np.array(self.state)

    def print_board(self):
        print(f"Number of bad attempts: {self.bad_tries}")
        print_state(self.state)

    # players are PLAYER1/PLAYER2 in order of turns
    def make_move(self, move, player):
        if move[0] > 15 or move[1] > 15:
            print("Please insert values between 0 and 14")
            return np.array(self.state).flatten(), -255, True
        elif not is_position_available(self.state, move):
            self.bad_tries += 1
            return np.array(self.state).flatten(), -10/255, False
        else:
            make_move(self.state, move, player)
            self.moves.append(move)
            self.round_number += 1
            self.winner = exists_winner(self.state, self.moves)
            state_array = np.array(self.state)
            state_array = state_array.flatten()
            if self.winner is EMPTY:
                self.turn = PLAYER2 if self.turn == PLAYER1 else PLAYER1
                return state_array, -1.0000/255.0000, False
            else:
                print(f"The winner is: {self.turn}")
                self.print_board()
                self.bad_tries = 0
                return state_array, 0, True


# Test game
if __name__ == '__main__':
    initial_state = get_initial_state()
    game = GomokuGame(initial_state)
    game.make_move([7, 7], PLAYER2)
    game.make_move([1, 8], PLAYER1)
    game.make_move([7, 8], PLAYER2)
    print(f"winner = {game.get_winner()}")

    game.make_move([7, 9], PLAYER2)
    game.make_move([2, 8], PLAYER1)
    game.make_move([7, 10], PLAYER2)
    game.make_move([2, 9], PLAYER1)
    game.make_move([7, 11], PLAYER2)
    print(game.make_move([1, 9], PLAYER1))

    print(f"winner = {game.get_winner()}")

    game.make_move([3, 11], PLAYER1)

    # game.print_board()
