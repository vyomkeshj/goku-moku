from env.checker import *
from env.gomoku import GomokuGame
from env.utils import get_initial_state, BOARD_SIZE


class GomokuEnv(gym.Env):

    def __init__(self):
        self.grid_size = BOARD_SIZE # 15x15 board
        self.action_count = 0
        self._max_episode_steps = BOARD_SIZE * BOARD_SIZE

        self.initial_state = get_initial_state()

        self.action_space = spaces.Discrete(self._max_episode_steps)

        self.observation_space = spaces.Box(0, 2, (self._max_episode_steps, ), dtype=np.int64)

        self.game_mdp = GomokuGame(self.initial_state)

    def step(self, action_step):
        row_index = int(action_step / self.grid_size)
        col_index = action_step % self.grid_size

        action_step = [row_index, col_index]
        player = self.game_mdp.get_current_turn()
        observation, reward, done = self.game_mdp.make_move(action_step, player)

        if self.action_count == self._max_episode_steps:
            done = True

        self.action_count += 1
        return observation, reward, done, {}

    def reset(self):
        self.initial_state = get_initial_state()
        self.game_mdp = GomokuGame(self.initial_state)
        self.action_count = 0

        # return empty board observation
        return np.zeros((BOARD_SIZE * BOARD_SIZE,), dtype=np.int8)

    def render(self, mode='human'):
        print("___________________GAME_RENDER_________________________")
        print(f"Rendering board after {self.action_count} steps")
        print(f"Winner = {self.game_mdp.get_winner()}")
        self.game_mdp.print_game_state()
        print("_________________________________________________")


# Test environment
if __name__ == '__main__':
    env = GomokuEnv()
    reset = env.reset()
    check_env(env)
