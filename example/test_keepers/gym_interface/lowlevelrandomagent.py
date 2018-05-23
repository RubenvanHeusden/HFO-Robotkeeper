from agent import Agent
from lowlevelactionspace import LowLevelActionSpace
import hfo
import helper
from neuralstatespace import NeuralStateSpace
from rewardfetcher import RewardFetcher

class LowLevelRandomAgent(Agent):
    def __init__(self, envir=hfo.HFOEnvironment(), action_set="low_level",seed=123):
        Agent.__init__(self,env=envir, agent_type="low_level_random_agent", action_set=LowLevelActionSpace(), 
        state_space=NeuralStateSpace(), feature_set=hfo.LOW_LEVEL_FEATURE_SET, port=6000,base="base_right", goalie=True)
        self.seed = seed        


    def episode(self):
        observation = self.reset()
        for t in range(1000):
            features = self.env.getState()
            helper.angle_feature(features)
            action = self.action_set.sample()
            observation, reward, done = self.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break
            #TODO : add a SERVER_DOWN case
            
    
