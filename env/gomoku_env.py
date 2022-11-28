from env.checker import *
from gym import spaces

from env.gomoku import GomokuGame
from env.utils import get_initial_state
from stable_baselines3 import A2C


class GomokuEnv(gym.Env):

    def __init__(self):
        self.action_count = 0
        self._max_episode_steps = 255

        self.initial_state = get_initial_state()

        self.action_space = spaces.Box(low=np.array([-1, -1]),
                                       high=np.array([1, 1]),
                                       dtype=np.float32)

        self.observation_space = spaces.Box(
            low=np.full((15, 15), -1, dtype=int),
            high=np.full((15, 15), 1, dtype=int),
            dtype=int
        )

        self.game_mdp = GomokuGame(self.initial_state)

        self.observation_buffer_size = 4000  # history size

    def step(self, action_step):
        """action is dict {player: PLAYER1/2, action: [x,y]}"""
        # print(f"old action: {action}")
        action_step = [int(x * 7.0000 + 7.0000) for x in action_step]
        # print(f"new action: {action}")

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
        # print(f"reset board after {self.action_count} steps")

        self.action_count = 0
        return self.game_mdp.get_board_state()

    def render(self, mode='human'):
        self.game_mdp.print_board()


# Test environment
if __name__ == '__main__':
    env = GomokuEnv()
    reset = env.reset()
    check_env(env)

    model = A2C("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=1000)

    obs = env.reset()
    for i in range(10):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()
