from environment import BaseEnvironment


class ADR_Environment(BaseEnvironment):
    def init(self):
        pass

    def env_init(self , env_info={}):
        pass

    def env_observe_state(self):
        pass

    def calculate_reward(self , state , action , next_state):
        pass

    def is_terminal(self , state):
        pass

    def perform_action(self , a):
        pass

    def env_start(self):
        pass

    def env_step(self, action):
        pass

    def env_cleanup(self):
        pass