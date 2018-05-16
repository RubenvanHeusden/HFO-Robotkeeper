from hfo import *
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import time
import argparse
import math
import threading

class hfo_agent(threading.Thread):
    def __init__(self, port, base, goalie):
        threading.Thread.__init__(self)
        self.port = port
        self.base = base
        self.goalie = goalie
        self.trial_number = 0
        self.step = 0
        self.last_step_heard = [-1, -1, -1, -1]
        self.s = None
        self.s1 = None
        self.a = -1
        self.r = None
        self.status = None
        self.saved = False
        self.say = ""
        self.hear = ""
        self.ep_history = []
        self.goals = []
        self.goals_in_row = 1
        self.hfo = HFOEnvironment()

    def run(self):
        self.connectToServer()
        self.status = IN_GAME

        self.getFirstState()
        while (self.status != SERVER_DOWN):
            # Get action from other thread
            self.say = "action"
            while self.hear != "action done":
                time.sleep(0.01)
            self.say = ""
            # Perform action
            self.perform_action(self.a)

            # Set step
            self.setStep()

            # Get reward
            self.getReward()

            # Update history
            self.updateHistory()

            # If end of trial update network in other thread
            if (self.status != IN_GAME):
                # Update network in other thread
                self.say = "update"
                while self.hear != "update done":
                    time.sleep(0.01)
                self.say = ""
                self.status = IN_GAME
                # Go to first state
                self.getFirstState()
                # Update trial_number
                self.trial_number += 1
            # Save the network in other thread
            if (self.trial_number % 1000 == 0 and self.saved == False):
                print("Saving the network")
                # Save the network in other thread
                self.say = "save"
                while self.hear != "save done":
                    time.sleep(0.01)
                self.say = ""
                self.saved = True
            # Check goals in a row and stop if scored ten times in a row
            if (self.goals_in_row == 10): break
            self.step += 1
            time.sleep(0.1)

        # Save the network in other thread
        self.say = "save"
        while self.hear != "save done":
            time.sleep(0.01)
        self.say = ""
        # Save evalution in other thread
        self.say = "stop"
        while self.hear != "stop done":
            time.sleep(0.01)
        self.say = ""
        self.hfo.act(QUIT)


    def connectToServer(self):
        print("Connect base: ", self.base)
        self.hfo.connectToServer(HIGH_LEVEL_FEATURE_SET,
                              'bin/teams/base/config/formations-dt', self.port,
                              'localhost', 'base_' + self.base, self.goalie)

    def get_distance_to_ball(self, states):
        return pow(pow((states[0] - states[3]), 2) + pow((states[1] - states[4]), 2), 0.5)

    def get_distance_ball_to_goal(self, states):
        return pow(pow((states[3] - 1), 2) + pow((states[4] - 0), 2), 0.5)

    def getFirstState(self):
        self.s = self.hfo.getState()
        self.s1 = self.s

        # Reset everything
        self.a = None
        self.r = None
        self.saved = False
        self.status = IN_GAME
        self.ep_history = []
        return self.s

    def getState(self):
        self.s1 = self.hfo.getState()
        return self.s1

    def setStep(self):
        self.status = self.hfo.step()
        return self.status

    def perform_action(self, a):
        self.a = a
        debug = False
        if (a == 0):
            # Walk
            if (debug):
                print("Walking")
            self.hfo.act(DASH, 100., 0.)
            return
        elif (a == 1):
            # Kick forward
            if (debug):
                print("Kicking")
            self.hfo.act(KICK, 100., 0.)
            return
        elif (a == 2):
            # Turn left
            if (debug):
                print("Turn left")
            self.hfo.act(TURN, -5.)
            return
        elif (a == 3):
            # Turn right
            if (debug):
                print("Turn right")
            self.hfo.act(TURN, 5.)
            return
        elif (a == 4):
            # Walk backward
            if (debug):
                print("Walk backward")
            self.hfo.act(DASH, -100., 0.)
            return
        elif (a == 5):
            # Walk left
            if (debug):
                print("Walk left")
            self.hfo.act(DASH, 100., -90.)
            return
        elif (a == 6):
            # Walk right
            if (debug):
                print("Walk right")
            self.hfo.act(DASH, 100., 90.)
            return
        elif (a == 7):
            # Kick left
            if (debug):
                print("Kick left")
            self.hfo.act(KICK, 100., -20.)
            return
        elif (a == 8):
            # Kick right
            if (debug):
                print("Kick right")
            self.hfo.act(KICK, 100., 20.)
            return
        elif (a == 9):
            # Soft kick forward
            if (debug):
                print("Soft kick forward")
            self.hfo.act(KICK, 5., 0.)
            return
        elif (a == 10):
            # Walk forward slowly
            if (debug):
                print("Walk forward slowly")
            self.hfo.act(DASH, 20., 0.)
            return
        elif (a == 11):
            # Stand (only for standing goalie)
            if (debug):
                print("Stand")
            self.hfo.act(DASH, 0., 0.)
            return

    def getReward(self):
        distance_to_ball = self.get_distance_to_ball(self.s1)
        distance_ball_to_goal = self.get_distance_ball_to_goal(self.s1)

        r = 0
        if (self.status == GOAL):
            r = 500
            self.goals.append(self.trial_number)
            print("Goals in a row: ", self.goals_in_row)
            if (len(self.goals) > 1 and self.goals[-2] == self.trial_number - 1):
                self.goals_in_row += 1
        elif (self.status == OUT_OF_TIME or self.status == CAPTURED_BY_DEFENSE or self.status == OUT_OF_BOUNDS):
            self.goals_in_row = 1
            r = -50

        if (distance_ball_to_goal < self.get_distance_ball_to_goal(self.s)):
            r += 1 - distance_ball_to_goal
        elif distance_ball_to_goal > self.get_distance_ball_to_goal(self.s):
            r += -1 * distance_ball_to_goal
        elif (distance_to_ball < self.get_distance_to_ball(self.s)):
            r += 1 - distance_to_ball
        else:
            r += -1 * distance_to_ball
        self.r = r

    def updateHistory(self):
        self.ep_history.append([self.s, self.a, self.r, self.s1])
        self.s = self.s1
