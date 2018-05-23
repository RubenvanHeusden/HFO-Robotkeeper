from actionset import ActionSet
import hfo
from statespace import StateSpace
from rewardfetcher import RewardFetcher
import numpy as np

''' 
This file contains the Agent class that is used in the goalkeeper experiments.
This implementation aims to function as a bridge between the HFO functions
and the openaigym implementation, so that it the methods conform to the function
and return values that the openaigym environment have.
'''


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
       
    def getStateSize(self):
        return self.env.getStateSize()
   
   
    
    def step(self, action):
        self.env.act(*self.action_set[action])
        status = self.env.step()
        if status == hfo.SERVER_DOWN:
            self.env.act(hfo.QUIT)
            exit()
        features = self.env.getState()
        new_state = self.state_space.get_state(features)
        reward = self.reward_fetcher.reward(features, status)
        done = not (status == hfo.IN_GAME)
   
        return new_state, reward, done
   
    def __str__(self):
        return "%s using %s" %(self.agent_type, self.action_set)
