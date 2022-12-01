from random import uniform

from env.checker import *
from env.gomoku import GomokuGame
from env.utils import get_initial_state, BOARD_SIZE, PLAYER, OPPONENT, EMPTY


class GomokuEnv(gym.Env):

    def __init__(self):
        self.grid_size = BOARD_SIZE    # 15x15 board
        self._max_episode_steps = BOARD_SIZE * BOARD_SIZE

        # The winning agent from the last match becomes the adversarial for the current epoch
        self.opponent = None

        # Who gets to play first
        self.turn = PLAYER if uniform(0, 1) > 0.5 else OPPONENT

        self.initial_state = get_initial_state()

        self.action_space = spaces.Discrete(self._max_episode_steps)

        self.observation_space = spaces.Box(0, 2, (self._max_episode_steps, ), dtype=np.int64)

        self.game_mdp = GomokuGame(self.initial_state)

    def set_opponent(self, opponent):
        self.opponent = opponent

    def step(self, action_step):
        row_index = int(action_step / self.grid_size)
        col_index = action_step % self.grid_size

        action_step_player = [row_index, col_index]
        if self.turn == PLAYER:

            # Make move for player
            observation_p, reward_p, done_p = self.game_mdp.make_move(action_step_player, PLAYER)

            if done_p:
                return observation_p, reward_p, True, {}

            # If more turns left or game not over, move for opponent and return this observation with player's reward
            action_step_opponent, _ = self.opponent.predict(observation_p, deterministic=False)
            observation_final, reward_p, done = self.game_mdp.make_move(action_step_opponent, OPPONENT)
        else:
            # Opponent's move first, with an empty board
            action_step_opponent, _ = self.opponent.predict(np.zeros(BOARD_SIZE*BOARD_SIZE), deterministic=False)
            observation_final, _, done_o = self.game_mdp.make_move(action_step_opponent, OPPONENT)

            if self.game_mdp.get_winner() == OPPONENT:
                # if the opponent has won, big negative reward
                return observation_final, -5, True, {}

            observation_final, reward_p, done = self.game_mdp.make_move(action_step_player, PLAYER)

        self.action_count += 1

        return observation_final, reward_p, done, {}

    def reset(self):
        self.initial_state = get_initial_state()
        self.game_mdp = GomokuGame(self.initial_state)

        # return empty board observation
        return np.zeros((BOARD_SIZE * BOARD_SIZE,), dtype=np.int8)

    def render(self, mode='human'):
        print("___________________GAME_RENDER_________________________")
        print(f"Rendering board after {self.game_mdp.get_current_move_count()} steps")
        self.game_mdp.print_game_state()
        print(f"Winner = {self.game_mdp.get_winner()}")
        print("_________________________________________________")


# Test environment
if __name__ == '__main__':
    env = GomokuEnv()
    reset = env.reset()
    check_env(env)
