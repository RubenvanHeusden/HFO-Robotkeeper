from __future__ import division
import hfo
import argparse
import random


def run(num_episodes):
    env= hfo.HFOEnvironment()
    r = 0

    env.connectToServer(hfo.LOW_LEVEL_FEATURE_SET,
                          '/home/student/Desktop/HFO-master_ruben/bin/teams/base/config/formations-dt', 6000,
                          'localhost', 'base_left', False)
                          
                          
    for episode in xrange(num_episodes): # replace with xrange(5) for Python 2.X
        kick_angle = random.randint(-30, 30)
        status = hfo.IN_GAME
        while status == hfo.IN_GAME:
            features = env.getState()
            env.act(hfo.KICK, 40.0, kick_angle)
            status = env.step()
        print 'Episode', episode, 'ended'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials',default=1000, type=int)
    args = parser.parse_args()
    run(num_episodes=args.trials)

if __name__ == "__main__":
    main()
