from actionspace import ActionSpace
import hfo

class HighLevelActionSpace(ActionSpace):
    def __init__(self):
        ActionSpace.__init__(self, action_set="high_level")
        self._action_list =  [tuple([hfo.DEFEND_GOAL]), tuple([hfo.INTERCEPT]), tuple([hfo.NOOP]), tuple([hfo.MOVE]), tuple([hfo.CATCH])]
