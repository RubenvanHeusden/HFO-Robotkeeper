from actionspace import ActionSpace
import hfo

class HighLevelActionSpace(ActionSpace):
    def __init__(self):
        ActionSpace.__init__(self, action_set="high_level")
        self._action_list =  [tuple([hfo.REDUCE_ANGLE_TO_GOAL]), tuple([hfo.INTERCEPT]), tuple([hfo.NOOP])]
