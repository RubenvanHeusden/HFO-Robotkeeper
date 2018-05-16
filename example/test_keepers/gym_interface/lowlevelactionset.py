from actionset import ActionSet
import hfo

class LowLevelActionSet(ActionSet):
    def __init__(self):
        ActionSet.__init__(self, action_set="low_level")
        self._action_list = [(hfo.DASH, 80, angle) for angle in [-180, -135, -90, -45, 0, 45, 90, 135]]+[tuple([hfo.NOOP])]
