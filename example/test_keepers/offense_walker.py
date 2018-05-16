from hfo import *
import helper
hfo = hfo.HFOEnvironment()
x_bounds = [-1, 1]
y_bounds = [-1, 1]
hfo.connectToServer(HIGH_LEVEL_FEATURE_SET,
                      '/home/student/Desktop/HFO-master_ruben/bin/teams/base/config/formations-dt', 6000,
                      'localhost', 'base_left', False)
for episode in xrange(5): # replace with xrange(5) for Python 2.X
  status = IN_GAME
  while status == IN_GAME:
    features = hfo.getState()
    helper.bin_ball_position(features[0:2], x_bounds, y_bounds)
    hfo.act(MOVE_TO, 0.0, 0.0)
    status = hfo.step()
  print 'Episode', episode, 'ended'
