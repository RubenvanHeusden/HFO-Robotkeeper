#!/usr/bin/env python
# encoding: utf-8

# Before running this program, first Start HFO server:
# $> ./bin/HFO --offense-agents 1

import itertools

from rl import rl
from hfo import *
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import time
import argparse
import math

gamma = 0.99
# Create the HFO Environment
hfo = HFOEnvironment()

def discount_rewards(r):
  """ take 1D float array of rewards and compute discounted reward """
  discounted_r = np.zeros_like(r)
  running_add = 0
  for t in reversed(xrange(0, r.size)):
      running_add = running_add * gamma + r[t]
      discounted_r[t] = running_add
  return discounted_r

def get_distance_to_ball(states):
  return pow(pow((states[0] - states[3]), 2) + pow((states[1] - states[4]), 2), 0.5)


def get_distance_ball_to_goal(states):
  return pow(pow((states[3] - 1), 2) + pow((states[4] - 0), 2), 0.5)

def compute_angle_to_ball(states):
    aanliggend = states[0] - states[3]
    overstaand = states[1] - states[4]
    theta = math.atan2(overstaand, aanliggend)
    return ((theta / math.pi)* 180.0) + 180

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

def perform_action(a):
    debug = False
    if (a == 0):
        # Walk
        if (debug):
            print("Walking")
        hfo.act(DASH, 100., 0.)
        return
    elif (a == 1):
        # Kick forward
        if (debug):
            print("Kicking")
        hfo.act(KICK, 100., 0.)
        return
    elif (a == 2):
        # Turn left
        if (debug):
            print("Turn left")
        hfo.act(TURN, -5.)
        return
    elif (a == 3):
        # Turn right
        if (debug):
            print("Turn right")
        hfo.act(TURN, 5.)
        return
    elif (a == 4):
        # Walk backward
        if (debug):
            print("Walk backward")
        hfo.act(DASH, -100., 0.)
        return
    elif (a == 5):
        # Walk left
        if (debug):
            print("Walk left")
        hfo.act(DASH, 100., -90.)
        return
    elif (a == 6):
        # Walk right
        if (debug):
            print("Walk right")
        hfo.act(DASH, 100., 90.)
        return
    elif (a == 7):
        # Kick left
        if (debug):
            print("Kick left")
        hfo.act(KICK, 100., -20.)
        return
    elif (a == 8):
        # Kick right
        if (debug):
            print("Kick right")
        hfo.act(KICK, 100., 20.)
        return
    elif (a == 9):
        # Soft kick forward
        if (debug):
            print("Soft kick forward")
        hfo.act(KICK, 5., 0.)
        return
    elif (a == 10):
        # Walk forward slowly
        if (debug):
            print("Walk forward slowly")
        hfo.act(DASH, 20., 0.)
        return

def test_model_left(args):
  print("Test base left")
  hfo.connectToServer(HIGH_LEVEL_FEATURE_SET,
                        'bin/teams/base/config/formations-dt', args.port,
                        'localhost', 'base_left', False)

  tf.reset_default_graph()  # Clear the Tensorflow graph.
  state_size = 11 + 6 * args.numTeammates + 3 * args.numOpponents
  myAgent = rl(lr=1e-2, s_size=state_size, a_size=11, h_size=8)  # Load the agent.

  saver = tf.train.Saver()

  with tf.Session() as sess:
    # Restore variables from disk
    saver.restore(sess, "models/" + args.modelFolder + "/final.ckpt")
    print("Model restored")

    goal = 0
    # Test 50 times
    for i in range(50):
      status = IN_GAME
      while status == IN_GAME:
        s = hfo.getState()
        a_dist = sess.run(myAgent.output, feed_dict={myAgent.state_in: [s]})
        a = np.argmax(a_dist[0])
        perform_action(a)
        status = hfo.step()

      if (status == GOAL):
        goal += 1

    print("Scored ", goal, "/", 50, " with an accuracy of ", goal/50.0)
    hfo.act(QUIT)

def train_base_right(args):
  # TODO: Move to own file
  hfo.connectToServer(HIGH_LEVEL_FEATURE_SET,
                        'bin/teams/base/config/formations-dt', args.port,
                        'localhost', 'base_left', False)
  print("Base right")

  status = IN_GAME
  while status != SERVER_DOWN:
      status = hfo.step()

def train_base_left(args):
  # Connect to the server with the specified
  # feature set. See feature sets in hfo.py/hfo.hpp.
  hfo.connectToServer(HIGH_LEVEL_FEATURE_SET,
                      'bin/teams/base/config/formations-dt', args.port,
                      'localhost', 'base_left', False)

  print("Base left")

  tf.reset_default_graph() #Clear the Tensorflow graph.
  state_size = 11 + 6 * args.numTeammates + 3 * args.numOpponents
  print state_size
  myAgent = rl(lr=1e-2,s_size=state_size,a_size=11,h_size=8) #Load the agent.

  total_episodes = 5000 #Set total number of episodes to train agent on.
  update_frequency = 5

  init = tf.global_variables_initializer()

  saver = tf.train.Saver()

  with tf.Session() as sess:
    sess.run(init)
    i = 0
    total_reward = []
    goals = []
    goals_in_row = 1

    gradBuffer = sess.run(tf.trainable_variables())
    for ix, grad in enumerate(gradBuffer):
      gradBuffer[ix] = grad * 0

    # while i < total_episodes:
    status = IN_GAME
    while status != SERVER_DOWN and goals_in_row < 10:
      s = hfo.getState()
      running_reward = 0
      ep_history = []
      status = IN_GAME
      step = 0
    #   if i == 0:
    #     # example
    #     while status == IN_GAME:
    #       theta = compute_angle_to_ball(s)
    #       print("Theta: ", theta)
    #       if s[6] == -1:
    #         # Walk to ball
    #         if (abs((theta + 180) % 360 - ((s[2] * 180.0) + 180)) > 7.):
    #             # Turn to the ball
    #             print("Turn to the ball")
    #             if (theta > 180):
    #                 a = 2
    #             else:
    #                 a = 3
    #         else:
    #             # Walk forward to the ball
    #             print("Walk to the ball")
    #             a = 0
    #       else:
    #         if (abs(s[2] - s[8]) > 0.01):
    #             # Turn to the goal
    #             print("Turn to the goal")
    #             if (theta > 180):
    #                 a = 2
    #             else:
    #                 a = 3
    #         else:
    #             # Dribble to goal
    #             print("Dribble to the goal")
    #             a = 9
    #       perform_action(a)
    #       status = hfo.step()
    #       s = hfo.getState()
    #     break
      while status == IN_GAME:
        if (i % 10 == 0):
             if s[6] == -1:
               theta = compute_angle_to_ball(s)
               print("Theta: ", theta)
               # Walk to ball
               if (abs((theta + 180) % 360 - ((s[2] * 180.0) + 180)) > 7.):
                   # Turn to the ball
                   print("Turn to the ball")
                   if ((theta + 180) % 360 - ((s[2] * 180.0) + 180) < 0):
                       a = 2
                   else:
                       a = 3
               else:
                   # Walk forward to the ball
                   print("Walk to the ball")
                   a = 0
             else:
               if (abs(s[2] - s[8]) > 0.01):
                   # Turn to the goal
                   print("Turn to the goal")
                   if (s[2] - s[8] > 0):
                       a = 2
                   else:
                       a = 3
               else:
                   # Dribble to goal
                   print("Dribble to the goal")
                   a = 9
        else:
            # Probabilistically pick an action given our network outputs.
            a_dist = sess.run(myAgent.output, feed_dict={myAgent.state_in: [s]})
            a = np.random.choice(a_dist[0], p=a_dist[0])
            a = np.argmax(a_dist == a)


        perform_action(a)
        status = hfo.step()
        s1 = hfo.getState()

        distance_to_ball = get_distance_to_ball(s1)
        distance_ball_to_goal = get_distance_ball_to_goal(s1)

        r = 0
        if (status == GOAL):
          r = 500
          goals.append(i)
          print("Goals in a row: ", goals_in_row)
          if (len(goals) > 1 and goals[-2] == i - 1):
            goals_in_row += 1
        elif (status == OUT_OF_TIME or status == CAPTURED_BY_DEFENSE or status == OUT_OF_BOUNDS):
          goals_in_row = 1
          r = -50

        if (distance_ball_to_goal < get_distance_ball_to_goal(s)):
          r += 1 - distance_ball_to_goal
        elif distance_ball_to_goal > get_distance_ball_to_goal(s):
          r += -1 * distance_ball_to_goal
        elif (distance_to_ball < get_distance_to_ball(s)):
          r += 1 - distance_to_ball
        else:
          r += -1 * distance_to_ball



        # TODO: distance to ball discreet maken
        # TODO: reward ball to goal
        # TODO: reward for kicking when close to the ball
        # TODO: actie voor dribbelen toevoegen


        ep_history.append([s, a, r, s1])
        s = s1
        running_reward += r
        step += 1
      print("Amount steps in one episode: ", step)
      print("Total reward one episode: ", running_reward)
      # Update the network.
      ep_history = np.array(ep_history)
      ep_history[:, 2] = discount_rewards(ep_history[:, 2])
      feed_dict = {myAgent.reward_holder: ep_history[:, 2],
                   myAgent.action_holder: ep_history[:, 1], myAgent.state_in: np.vstack(ep_history[:, 0])}
      grads = sess.run(myAgent.gradients, feed_dict=feed_dict)
      for idx, grad in enumerate(grads):
        gradBuffer[idx] += grad

      if i % update_frequency == 0 and i != 0:
        feed_dict = dictionary = dict(zip(myAgent.gradient_holders, gradBuffer))
        _ = sess.run(myAgent.update_batch, feed_dict=feed_dict)
        for ix, grad in enumerate(gradBuffer):
          gradBuffer[ix] = grad * 0

      total_reward.append(running_reward)

      # Update our running tally of scores.
      if i % 1000 == 0:
        print(np.mean(total_reward[-1000:]))
        save_path = saver.save(sess, "models/" + args.modelFolder + "/" + str(i) + ".ckpt")
        print("Model saved in file: %s" % save_path)
      i += 1
      time.sleep(0.1)
    print("Total amount of episodes: ", i)
    save_path = saver.save(sess, "models/" + args.modelFolder + "/final.ckpt")
    print("Model saved in file: %s" % save_path)
    write_result_to_file(i, goals, a, args.modelFolder)
    hfo.act(QUIT)

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--port', type=int, default=6000)
  parser.add_argument('--numTeammates', type=int, default=0)
  parser.add_argument('--numOpponents', type=int, default=0)
  parser.add_argument('--base', type=str, default='left')
  parser.add_argument('--modelFolder', type=str, default="test_model")
  parser.add_argument('--test', type=bool, default=False)
  args=parser.parse_args()

  if (args.base == 'left'):
    if (args.test):
      test_model_left(args)
    else:
      train_base_left(args)
  else:
    train_base_right(args)


if __name__ == '__main__':
  main()
