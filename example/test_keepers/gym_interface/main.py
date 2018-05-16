from lowlevelrandomagent import LowLevelRandomAgent
import tensorflow as tf

agent = LowLevelRandomAgent()



for i_episode in range(100):
    agent.episode()
