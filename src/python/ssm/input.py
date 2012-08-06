import numpy as np
from scipy.stats import expon

SPIKE_HISTORY_LENGTH = 20 #don't store more than this amount of spikes

class SpikingInputStream(object):

    def __init__(self):
        self.id = None
        self.length = 0

    def __len__(self):
        return self.length

    def get_spikes(self):
        pass

class PoissonCurrentStream(SpikingInputStream):
    """ Uses time rescaling to transform a continuous input into a spike train. """

    def __init__(self, rate_func, dt):
        SpikingInputStream.__init__(self)
        self.dt = dt
        self.length = 1
        self.rate_func = rate_func
        self.spikes = np.ones([1, SPIKE_HISTORY_LENGTH])*-10000.0
        self.rv_expon = expon()
        self.reset()

    def reset(self):
        self.cumsum = 0.0
        self.next_spike_time = self.rv_expon.rvs()

    def step(self, t):
        self.cumsum += self.rate_func(t) * self.dt
        if self.cumsum >= self.next_spike_time:
            #print 'PoissonCurrentStream: spike at %0.6f' % t
            self.spikes[0, :-1] = self.spikes[0, 1:]
            self.spikes[0, -1] = t
            self.reset()

    def get_spikes(self):
        return self.spikes




