from env.checker import *
from gym import spaces

from env.gomoku import GomokuGame
from env.utils import get_initial_state


class GomokuEnv(gym.Env):

    def __init__(self):
        self.action_count = 0
        self._max_episode_steps = 255

        self.initial_state = get_initial_state()

        self.action_space = spaces.Box(low=np.array([-1, -1]),
                                       high=np.array([1, 1]),
                                       dtype=np.float32)

        self.observation_space = spaces.Box(
            low=np.full((225,), 0, dtype=np.uint8),
            high=np.full((225,), 255, dtype=np.uint8),
            dtype=int
        )

        self.game_mdp = GomokuGame(self.initial_state)

    def step(self, action_step):
        """action is dict {player: PLAYER1/2, action: [x,y]}"""
        action_step = [int(x * 7.0000 + 7.0000) for x in action_step]
        # print(f" action: {action}")

        self.action_count += 1
        # print("action done = ",action)
        player = self.game_mdp.get_current_turn()
        observation, reward_received, done_status = self.game_mdp.make_move(action_step, player)

        if self.action_count == self._max_episode_steps:
            done_status = True
        return observation, reward_received, done_status, {}

    def reset(self):
        self.initial_state = get_initial_state()
        self.game_mdp = GomokuGame(self.initial_state)
        print(f"reset board after {self.action_count} steps")

        self.action_count = 0
        observation = self.game_mdp.get_board_state().flatten()
        return observation

    def render(self, mode='human'):
        # pass
        self.game_mdp.print_board()


# Test environment
if __name__ == '__main__':
    env = GomokuEnv()
    reset = env.reset()
    check_env(env)
