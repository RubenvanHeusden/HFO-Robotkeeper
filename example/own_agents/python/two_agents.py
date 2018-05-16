#!/usr/bin/env python
# encoding: utf-8

from rl import rl
from hfo import *
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import time
import argparse
import math
import thread
from hfo_agent import hfo_agent

gamma = 0.99

def discount_rewards(r):
  """ take 1D float array of rewards and compute discounted reward """
  discounted_r = np.zeros_like(r)
  running_add = 0
  for t in reversed(xrange(0, r.size)):
      running_add = running_add * gamma + r[t]
      discounted_r[t] = running_add
  return discounted_r


def write_result_to_file(trials, goals, last_action, folder):
  file_object = open("models/" + folder + "/evaluation", "w")
  file_object.write("Amount of trials: \n")
  file_object.write(str(trials))
  file_object.write("\nScored in episodes: \n")
  file_object.write("[")
  for goal in goals:
    file_object.write(str(goal) + ", ")
  file_object.write("]\n")
  file_object.write("Last action: \n")
  file_object.write(str(last_action))

  file_object.close()

def train(args):
    hfo_agent1 = hfo_agent(args.port, args.base, False)
    hfo_agent2 = hfo_agent(args.port, "right", True)
    hfo_agent1.start()
    hfo_agent2.start()

    print("Started the thread")
    agents = [hfo_agent1, hfo_agent2]

    # Set up tensorflow rl
    tf.reset_default_graph() #Clear the Tensorflow graph.
    state_size = 11 + 6 * args.numTeammates + 3 * args.numOpponents
    myAgent = rl(lr=1e-2,s_size=state_size,a_size=11,h_size=8) #Load the rl.
    update_frequency = 5
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        gradBuffer = sess.run(tf.trainable_variables())
        for ix, grad in enumerate(gradBuffer):
            gradBuffer[ix] = grad * 0

        # Start listening to the agents
        while (True):
            for agent in agents:
                if (agent.base == "left"):
                    if (agent.say == "action"):
                        if (agent.last_step_heard[0] != agent.step):
                            agent.last_step_heard[0] = agent.step
                            # Select action
                            # Probabilistically pick an action given our network outputs.
                            a_dist = sess.run(myAgent.output, feed_dict={myAgent.state_in: [agent.s]})
                            a = np.random.choice(a_dist[0], p=a_dist[0])
                            a = np.argmax(a_dist == a)
                            agent.a = a
                        agent.hear = "action done"
                    elif (agent.say == "update" and len(agent.ep_history) > 0):
                        if (agent.last_step_heard[1] != agent.step):
                            agent.last_step_heard[1] = agent.step
                            # Update the network
                            ep_history = np.array(agent.ep_history)
                            if (len(agent.ep_history) > 0):
                                ep_history[:, 2] = discount_rewards(ep_history[:, 2])
                                feed_dict = {myAgent.reward_holder: ep_history[:, 2],
                                             myAgent.action_holder: ep_history[:, 1], myAgent.state_in: np.vstack(ep_history[:, 0])}
                                grads = sess.run(myAgent.gradients, feed_dict=feed_dict)
                                for idx, grad in enumerate(grads):
                                    gradBuffer[idx] += grad

                                if (agent.trial_number % update_frequency == 0 and agent.trial_number != 0):
                                    feed_dict = dictionary = dict(zip(myAgent.gradient_holders, gradBuffer))
                                    _ = sess.run(myAgent.update_batch, feed_dict=feed_dict)
                                    for ix, grad in enumerate(gradBuffer):
                                        gradBuffer[ix] = grad * 0
                        agent.hear = "update done"
                    elif (agent.say == "save"):
                        if (agent.last_step_heard[2] != agent.step):
                            agent.last_step_heard[2] = agent.step
                            save_path = saver.save(sess, "models/" + args.modelFolder + "/" + str(agent.trial_number) + ".ckpt")
                            print("Model saved in file: %s" % save_path)
                        agent.hear = "save done"
                    elif (agent.say == "stop"):
                        if (agent.last_step_heard[3] != agent.step):
                            agent.last_step_heard[3] = agent.step
                            print("Total amount of trials: ", agent.trial_number)
                            save_path = saver.save(sess, "models/" + args.modelFolder + "/final.ckpt")
                            print("Model saved in file: %s" % save_path)
                            write_result_to_file(agent.trial_number, agent.goals, agent.a, args.modelFolder)
                        agent.hear("stop done")
                else:
                    if (agent.say == "action"):
                        agent.a = 11
                        agent.hear = "action done"
                    elif (agent.say == "update"):
                        # Don't update
                        agent.hear = "update done"
                    elif (agent.say == "save"):
                        # Don't save
                        agent.hear = "save done"
                    elif (agent.say == "stop"):
                        # Don't stop
                        agent.hear = "stop done"


def train2(args):
  hfo_agent1 = hfo_agent(args.port, args.base, False)
  hfo_agent2 = hfo_agent(args.port, "right", True)
  hfo_agent1.start()
  hfo_agent2.start()

  while (hfo_agent1.status != IN_GAME and hfo_agent2.status != IN_GAME):
      print("Waiting for connection")

  time.sleep(10)

  tf.reset_default_graph() #Clear the Tensorflow graph.
  state_size = 11 + 6 * args.numTeammates + 3 * args.numOpponents
  myAgent = rl(lr=1e-2,s_size=state_size,a_size=11,h_size=8) #Load the rl.

  update_frequency = 5

  init = tf.global_variables_initializer()

  saver = tf.train.Saver()

  with tf.Session() as sess:
    sess.run(init)
    trial_number = 0

    gradBuffer = sess.run(tf.trainable_variables())
    for ix, grad in enumerate(gradBuffer):
      gradBuffer[ix] = grad * 0

    status = IN_GAME
    while status != SERVER_DOWN: # and goals_in_row < 10:
      a1_s = hfo_agent1.getFirstState()
      a2_s = hfo_agent2.getFirstState()

      status = IN_GAME
      while status == IN_GAME:
        # Action hfo1
        # Probabilistically pick an action given our network outputs.
        a1_a_dist = sess.run(myAgent.output, feed_dict={myAgent.state_in: [a1_s]})
        a1_a = np.random.choice(a1_a_dist[0], p=a1_a_dist[0])
        a1_a = np.argmax(a1_a_dist == a1_a)
        hfo_agent1.perform_action(a1_a)
        # Action hfo2
        # a2_a =
        # hfo_agent2.perform_action(a2_a)

        # Set step hfo1
        a1_status = hfo_agent1.setStep()
        a1_s1 = hfo_agent1.getState()
        # Set step hfo2
        a2_status = hfo_agent2.setStep()
        a2_s1 = hfo_agent2.getState()

        # Get reward hfo1
        hfo_agent1.getReward(trial_number)
        # Get reward hfo2
        # hfo_agent2.getReward(trial_number)

        # Set to next state and update history hfo1
        a1_s = a1_s1
        hfo_agent1.updateHistory()
        # Set to next state and update history hfo1
        # a2_s = a2_s1
        # hfo_agent2.updateHistory()

        if (a1_status != IN_GAME or a2_status != IN_GAME):
          if (a1_status == SERVER_DOWN or a2_status == SERVER_DOWN):
              status = SERVER_DOWN
          else:
              status = a1_status # TODO: maakt volgens mij niet uit wat deze krijgt

      # Update the network hfo1
      a1_ep_history = np.array(hfo_agent1.ep_history)
      a1_ep_history[:, 2] = discount_rewards(a1_ep_history[:, 2])
      a1_feed_dict = {myAgent.reward_holder: a1_ep_history[:, 2],
                   myAgent.action_holder: a1_ep_history[:, 1], myAgent.state_in: np.vstack(a1_ep_history[:, 0])}
      grads = sess.run(myAgent.gradients, feed_dict=a1_feed_dict)
      for idx, grad in enumerate(grads):
        gradBuffer[idx] += grad

      if trial_number % update_frequency == 0 and trial_number != 0:
        a1_feed_dict = dictionary = dict(zip(myAgent.gradient_holders, gradBuffer))
        _ = sess.run(myAgent.update_batch, feed_dict=a1_feed_dict)
        for ix, grad in enumerate(gradBuffer):
          gradBuffer[ix] = grad * 0

      # Update the network hfo2
      # TODO: update

      # Save the network
      if trial_number % 1000 == 0:
        save_path = saver.save(sess, "models/" + args.modelFolder + "/" + str(trial_number) + ".ckpt")
        print("Model saved in file: %s" % save_path)
      trial_number += 1

      # Check goals in a row
      if (hfo_agent1.goals_in_row == 10 or hfo_agent2.goals_in_row == 10): break
    print("Total amount of trials: ", trial_number)
    save_path = saver.save(sess, "models/" + args.modelFolder + "/final.ckpt")
    print("Model saved in file: %s" % save_path)
    write_result_to_file(trial_number, hfo_agent1.goals, a1_a, args.modelFolder)
    hfo_agent1.hfo.act(QUIT)
    hfo_agent2.hfo.act(QUIT)

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--port', type=int, default=6000)
  parser.add_argument('--numTeammates', type=int, default=0)
  parser.add_argument('--numOpponents', type=int, default=0)
  parser.add_argument('--base', type=str, default='left')
  parser.add_argument('--modelFolder', type=str, default="test_model")
  parser.add_argument('--test', type=bool, default=False)
  args=parser.parse_args()

  train(args)


if __name__ == '__main__':
  main()
