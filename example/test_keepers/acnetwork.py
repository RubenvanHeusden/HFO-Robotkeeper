import argparse
import hfo
import tensorflow as tf
import numpy as np
from gym_interface.lowlevelrandomagent import LowLevelRandomAgent
from actor import Actor
from critic import Critic
from gym_interface.memory import Memory


def get_param_from_action(a_index, params, action_space):
    a_index = np.argmax(a_index, 0)
    pars = [len(item[1:]) for item in action_space]        
    index_start = sum(pars[:a_index])
    index_end = index_start + pars[a_index]
    return a_index, params[index_start:index_end]


def run(num_episodes, buf_size):
    # initializing the parameters for the AC network
    e = 0.1
    total_steps = 0
    pre_train_steps = 1000
    update_freq = 500
    batch_size = 32
    
    # initializing the HFO environment
    env = LowLevelRandomAgent()
    state_space_dim = env.getStateSize()
    action_space_dim = env.action_space.n
    
    
    # initializing the experienc_replay buffer
    exp_buf = Memory(buf_size)
    
    
    with tf.Session() as sess:
        
        # initializing the actor and the critic networks
        actor = Actor(sess, num_inputs = state_space_dim, 
                                num_actions=action_space_dim, 
                                num_params=env.action_space.params)        
        
        critic = Critic(sess, state_size=state_space_dim, 
                        action_size=action_space_dim, 
                        param_size=env.action_space.params, num_outputs=1)

        # calling the variable initter AFTER the init of the networks
        init = tf.global_variables_initializer()
        sess.run(init)

        
        #TODO : CHANGED PREDICT TO OUTPUT NOT ARGMAX BUT ALL ACTIONS
        # DO THE MAX FUCNTION SOMEWHERE HERE
        
        # setting the initial variables for the HFO environments
        for x in range(num_episodes):
            total_reward = 0
            s = env.reset()
            done = False
            t_buf = Memory(100)
            
            for x in range(100):
                #TODO : random action according to boltzmann thin in Caitlin's paper
                #TODO : something with random noise (see code about DDPG)
                actions, params = actor.predict(s)
                action, best_params = get_param_from_action(actions[0], params[0], 
                                                    env.action_space)
                
                #TODO coubts about wether this return what I expect
                best_action = env.action_space[action]
                best_action[1:] = best_params
                s1, reward, done = env.step_raw(best_action)
                t_buf.add([s, action, reward, s1, done])
                
                # update the network with batches of experience
                if total_steps > pre_train_steps:
                    if total_steps % update_freq == 0:
                        
                        # retrieving batches of experiences from the memory buffer
                        update_batch = np.array(exp_buf.get_batch(batch_size)).reshape((batch_size, 5))
                        actor_inputs = np.vstack(update_batch[:, 0])
                        actor_outputs = actor.predict(actor_inputs)
                        c_actor_outputs = tf.concat((actor_outputs[0], actor_outputs[1]), axis=1)
                        critic_inputs = tf.concat((actor_inputs, c_actor_outputs), axis=1)
                        critic_output = critic.predict(critic_inputs.eval())
                        print critic.get_gradients(critic_inputs.eval(), 
                        actor_inputs, actor_outputs[0], actor_outputs[1])
                
                if done:
                    break
                
                s = s1    
                total_steps+=1
            for item in t_buf._data:
                exp_buf.add(item)      


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials', default=1000, type=int)
    parser.add_argument('--port', default=6000, type=int)
    parser.add_argument('--buffer_size', default=10000, type=int)
    args = parser.parse_args()
    
    # run the main program
    run(num_episodes=args.trials,buf_size=args.buffer_size)
    

if __name__ == "__main__":
    main()
