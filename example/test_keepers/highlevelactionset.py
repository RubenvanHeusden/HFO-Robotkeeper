from actionset Import ActionSet
import hfo

class HighLevelActionSet(ActionSet):
    def __init__(self):
        ActionSet.__init__(self, action_set="high_level")
        self._action_list = [tuple([hfo.CATCH]), tuple([hfo.REDUCE_ANGLE_TO_GOAL]), tuple([hfo.INTERCEPT]), tuple([hfo.GO_TO_BALL]), tuple([hfo.NOOP])]
