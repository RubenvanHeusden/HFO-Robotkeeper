from actionset import ActionSet
import hfo

class HighLevelActionSet(ActionSet):
    def __init__(self):
        ActionSet.__init__(self, action_set="high_level")
        self._action_list =  [tuple([hfo.REDUCE_ANGLE_TO_GOAL]), tuple([hfo.INTERCEPT]), tuple([hfo.NOOP])]
