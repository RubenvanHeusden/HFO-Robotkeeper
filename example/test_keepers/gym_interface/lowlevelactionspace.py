from actionspace import ActionSpace
import hfo

class LowLevelActionSpace(ActionSpace):
    def __init__(self):
        ActionSpace.__init__(self, action_set="low_level")
        self._action_list = [(hfo.DASH, 0.0, 0.0), (hfo.TACKLE, 0.0), (hfo.TURN, 0.0)] 
        self.params = sum(len(item[1:]) for item in self._action_list)
        self.n = len(self._action_list)
        # hfo.NOOP
        #self._action_list = [(hfo.DASH, 70.0, angle) for angle in [-180, -135, -90, -45, 0, 45, 90, 135]]+[tuple([hfo.NOOP])]
