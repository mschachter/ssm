
import numpy as np
from scipy.sparse import lil_matrix
import matplotlib.pyplot as plt
from ssm.input import PoissonCurrentStream, SPIKE_HISTORY_LENGTH


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

        self.num_input_streams = {} # number of presynaptic input streams
        self.num_input_neurons = {} # number of presynaptic neurons
        self.input_neuron_taus = {} # synapse_tau for each input neuron, key is post-synaptic neuron id

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

        #initialize neuron input counts
        for ssm in self.neurons:
            self.num_input_neurons[ssm.id] = 0
            self.num_input_streams[ssm.id] = 0

        #set up sparse connection weight matrix
        self.Wlil = lil_matrix((N, N))
        for (pre_id,post_id,w) in self.connections:
            self.Wlil[pre_id, post_id] = w
            self.num_input_neurons[post_id] += 1
        self.W = self.Wlil.tocsc() #more efficient sparse matrix for column access

        #set up sparse input connectivity matrices
        self.Win = {}
        for stream in self.streams:
            self.Win[stream.id] = lil_matrix((len(stream), N))
        for stream_id, stream_index, neuron_id, weight in self.stream_connections:
            self.Win[stream_id][stream_index, neuron_id] = weight
            self.num_input_streams[neuron_id] += 1
        for stream_id in self.Win.keys():
            self.Win[stream_id] = self.Win[stream_id].tocsc() #more efficient sparse matrix for column access

        #set up synapse taus for presynaptic neurons
        for ssmn in self.neurons:

            nin = self.num_input_streams[ssmn.id]
            nz = self.W[:, ssmn.id].nonzero()[0]
            itaus = np.zeros([nin + len(nz)])

            #input streams have neuron's default tau
            itaus[:nin] = ssmn.synapse_tau
            #each input neuron could have a different synapse tau
            for k,pre_id in enumerate(nz):
                itaus[nin+k] = self.neurons[pre_id].synapse_tau
            self.input_neuron_taus[ssmn.id] = itaus

    def step(self):
        t = self.tn * self.step_size
        rnums = np.random.random(len(self.neurons))

        #get next stream state
        stream_spikes = {}
        for stream in self.streams:
            stream.step(t)
            stream_spikes[stream.id] = stream.get_spikes()

        #update neural state
        for k,ssmn in enumerate(self.neurons):

            #gather up input spikes and weights to this neuron
            num_inputs = self.num_input_streams[ssmn.id] + self.num_input_neurons[ssmn.id]
            ispikes = np.zeros([num_inputs, SPIKE_HISTORY_LENGTH])
            iwts = np.zeros([num_inputs])

            nindex = ssmn.id
            input_index = 0
            #compute input spikes for this neuron
            for stream in self.streams:
                Ws = self.Win[stream.id]
                nz = Ws[:, nindex].nonzero()[0]
                ss = stream_spikes[stream.id][nz]
                ispikes[input_index:len(nz), :] = ss
                #print 'idx=%d, stream.id=%d, Ws=' % (idx, stream.id)
                #print Ws.todense()
                iwts[input_index:len(nz)] = Ws[nz, nindex].toarray().squeeze()
                input_index += len(nz)

            #compute membrane potential and refractory state
            pre_indices = self.W[:, nindex].nonzero()[0]
            if len(pre_indices) > 0:
                eindex = input_index + len(pre_indices)
                ispikes[input_index:eindex, :] = self.spikes[pre_indices, :]
                iwts[input_index:eindex] = self.W[pre_indices, nindex]
                input_index = eindex

            u = ssmn.u(t, ispikes, iwts, synapse_tau=self.input_neuron_taus[ssmn.id])
            g = ssmn.g(u)
            R = ssmn.R(t, self.spikes[nindex, -1])
            p = ssmn.pspike(g, R, self.step_size)

            #determine spike state
            spike = False
            if rnums[nindex] < p:
                spike = True
                self.spikes[nindex, :-1] = self.spikes[nindex, 1:]
                self.spikes[nindex, -1] = t

            self.state[nindex, 0] = u
            self.state[nindex, 1] = g
            self.state[nindex, 2] = R
            self.state[nindex, 3] = p
            self.state[nindex, 4] = spike

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

    def R(self, t, last_spike_time=-10000.0):
        """ Refractory function """
        if last_spike_time > 0.0:
            dt = t - last_spike_time - self.tau_abs
            #print 'last_spike_time=%0.4f, dt=%0.4f, tau_abs=%0.3f, tau_ref=%0.3f' % (last_spike_time, dt, self.tau_abs, self.tau_ref)
            if dt > 0.0:
                return dt**2 / (self.tau_ref**2 + dt**2)
            else:
                return 0.0
        return 1.0

    def u(self, t, pre_spike_times, w, synapse_tau=None):
        """ Computes membrane potential for this cell based on the spike history of the presynaptic
            cells and synaptic weight vector w. pre_spike_times has shape=NxSPIKE_HISTORY_LENGTH and
            w is of length N, where N are the # of presynaptic inputs.
        """

        #print 'pre_spike_times:'
        #print pre_spike_times
        #print 'w:'
        #print w
        if synapse_tau is None:
            synapse_tau = self.synapse_tau

        dt = t - pre_spike_times
        synaptic_currents = self.eps(dt, synapse_tau)
        weighted_current = np.dot(synaptic_currents.transpose(), w).sum()
        return self.u0 + weighted_current

    def eps(self, dt, synapse_tau):
        """ Exponentially decaying synaptic current as function of difference between current time and last spike time.
            dt can be a matrix as long as the length if synapse_tau is either 1 or equal to the number of rows in dt.
         """
        return np.exp(-dt / synapse_tau)

    def pspike(self, g, R, dt):
        """ Computes probability of spike at time t """
        return 1.0 - np.exp(-g*R*dt)


