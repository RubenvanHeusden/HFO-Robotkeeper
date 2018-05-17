from actionset import ActionSet
import hfo
from statespace import StateSpace
from rewardfetcher import RewardFetcher

class Agent:
    def connect(self, env, feature_set, port, base, goalie):
        env.connectToServer(feature_set,
                      '/home/student/Desktop/HFO-master_ruben/bin/teams/base/config/formations-dt', port,
                      'localhost', base, goalie)
        
    
    def __init__(self, env, agent_type, action_set, state_space, feature_set, port, base, goalie):
        self.agent_type = agent_type
        self.env = env
        self.action_set = action_set
        self.connect(env, feature_set, port, base, goalie)
        self.state_space = state_space
        self.reward_fetcher = RewardFetcher()
        
    def episode(self):
        """function handle to be overridden by subclassess"""
        pass
        
    def reset(self):
        features = self.env.getState()
        observation = self.state_space.get_state(features)
        return observation
    
    def step(self, action):
        self.env.act(*self.action_set[action])
        status = self.env.step()
        if status == hfo.SERVER_DOWN:
            self.env.act(hfo.QUIT)
            exit()
        new_state = self.env.getState()
        observation = self.state_space.get_state(new_state)
        reward = self.reward_fetcher.reward(new_state, status)
        done = not (status == hfo.IN_GAME)
   
        return observation, reward, done
   
    def __str__(self):
        return "%s using %s" %(self.agent_type, self.action_set)
