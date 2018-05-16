from agent import Agent
from highlevelactionset import HighLevelActionSet
import hfo
import helper
from statespace import StateSpace
from rewardfetcher import RewardFetcher

class HighLevelAgent(Agent):
    def __init__(self, envir=hfo.HFOEnvironment(), action_set="high_level",seed=123):
        Agent.__init__(self,env=envir, agent_type="high_level_agent", action_set=HighLevelActionSet(), 
        state_space=StateSpace(500), feature_set=hfo.HIGH_LEVEL_FEATURE_SET, port=6000,base="base_right", goalie=True)
        self.seed = seed        

    def episode(self):
        observation = self.reset()
        for t in range(1000):
            action = self.action_set.sample()
            observation, reward, done = self.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break
            #TODO : add a SERVER_DOWN case
    
    
