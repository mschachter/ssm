
import numpy as np
from scipy.sparse import lil_matrix
import matplotlib.pyplot as plt
from ssm.input import InputCurrentStream

SPIKE_HISTORY_LENGTH = 10 #don't store more than this amount of spikes

class SSMNetwork(object):

    def __init__(self, step_size=0.5e-3):
        self.neurons = []   #list of neurons
        self.step_size = step_size
        self.connections = []

        self.streams = []
        self.stream_connections = []

        self.spikes = None  #matrix of spike times for each neuron
        self.state = None   #(u, g, R) for each neuron

        self.tn = 0 #discrete time step number
        self.W = None
        self.Win = None

    def add(self, ssm_neuron):
        ssm_neuron.id = len(self.neurons)
        self.neurons.append(ssm_neuron)

    def connect(self, id1, id2, weight):
        self.connections.append( (id1, id2, weight) )

    def add_stream(self, stream):
        stream.id = len(self.streams)
        self.streams.append(stream)

    def connect_stream(self, stream_id, stream_index, neuron_id, weight):
        self.stream_connections.append( (stream_id, stream_index, neuron_id, weight))


    def compile(self):
        N = len(self.neurons)
        #set up spike history
        self.spikes = np.ones([N, SPIKE_HISTORY_LENGTH])*-100000.0
        #set up state
        self.state = np.zeros([N, 5])

        #set up sparse connection weight matrix
        self.Wlil = lil_matrix((N, N))
        for (pre_id,post_id,w) in self.connections:
            self.Wlil[pre_id, post_id] = w
        self.W = self.Wlil.tocsc() #more efficient sparse matrix for column access

        #set up sparse input connectivity matrices
        self.Win = {}
        for stream in self.streams:
            self.Win[stream.id] = lil_matrix((len(stream), N))
        for stream_id, stream_index, neuron_id, weight in self.stream_connections:
            self.Win[stream_id][stream_index, neuron_id] = weight
        for stream_id in self.Win.keys():
            self.Win[stream_id] = self.Win[stream_id].tocsc() #more efficient sparse matrix for column access


    def step(self):
        t = self.tn * self.step_size
        rnums = np.random.random(len(self.neurons))

        #get next stream state
        stream_states = {}
        for stream in self.streams:
            stream_states[stream.id] = stream.next(t)

        #update neural state
        for k,ssmn in enumerate(self.neurons):

            idx = ssmn.id
            #compute input current for this neuron
            input_current = 0.0
            for stream in self.streams:
                Ws = self.Win[stream.id]
                nz = Ws[:, idx].nonzero()[0]
                ss = stream_states[stream.id][nz]
                input_current += np.dot(Ws[nz, idx].toarray().squeeze(), ss)

            #compute membrane potential and refractory state
            pre_indices = self.W[:, idx].nonzero()[0]
            u = ssmn.u(t, self.spikes[pre_indices, :], self.W[pre_indices, idx])
            g = ssmn.g(u)
            R = ssmn.R(t)
            p = ssmn.pspike(g, R, self.step_size)

            #determine spike state
            spike = False
            if rnums[idx] < p:
                spike = True
                self.spikes[idx, :-1] = self.spikes[idx, 1:]
                self.spikes[idx, -1] = t

            self.state[idx, 0] = u
            self.state[idx, 1] = g
            self.state[idx, 2] = R
            self.state[idx, 3] = p
            self.state[idx, 4] = spike

        self.tn += 1


class SSMNeuron(object):

    def __init__(self):

        self.id = None
        self.r0 = 11.0           #average firing rate
        self.u0 = -0.065         #resting membrane potential
        self.du = 0.002          #?
        self.tau_abs = 0.003     #absolute refractory period
        self.tau_ref = 0.010     #overall refractory time
        self.synapse_tau = 0.010 #time constant of exponential synaptic decay

    def g(self, u):
        """ Gain function """
        gu = self.r0 * np.log(1.0 + np.exp( (u - self.u0) / self.du ) )
        return gu

    def R(self, t):
        """ Refractory function """
        if len(self.spikes) > 0:
            that = self.spikes[-1]
            dt = t - that - self.tau_abs
            if dt > 0.0:
                return dt**2 / (self.tau_ref**2 + dt**2)
        return 1.0

    def u(self, t, pre_spike_times, w, input_current=0.0):
        """ Computes membrane potential for this cell based on the spike history of the presynaptic
            cells and synaptic weight vector w.
        """
        synaptic_current = 0.0
        for j,stj in enumerate(pre_spike_times):
            scj = 0.0 #total synaptic current from cell j
            for tj in stj:
                scj += self.eps(t - tj)
            synaptic_current += w[j] * synaptic_current
        return self.u0 + synaptic_current + input_current

    def eps(self, dt):
        """ Exponentially decaying synaptic current as function of difference between current time and last spike time """
        return np.exp(-dt / self.synapse_tau)

    def pspike(self, g, R, dt):
        """ Computes probability of spike at time t """
        return 1.0 - np.exp(-g*R*dt)


def test_neuron(duration=0.100, step_size=0.5e-3):

    net = SSMNetwork(step_size=step_size)
    ssm = SSMNeuron()
    net.add(ssm)

    def ifunc(t):
        if t > 0.020:
            return 1.0
        return 0.0
    ics = InputCurrentStream(ifunc)
    net.add_stream(ics)
    net.connect_stream(ics.id, 0, ssm.id, 0.5)
    net.compile()

    nsteps = int(duration / step_size)
    t = np.arange(nsteps)*step_size

    state = np.zeros([nsteps, 1, 5])
    for k,ti in enumerate(t):
        state[k, 0, :] = net.state
        net.step()








