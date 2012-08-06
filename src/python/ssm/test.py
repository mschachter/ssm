import copy
import numpy as np
import matplotlib.pyplot as plt
import time

from ssm.input import PoissonCurrentStream
from ssm.network import SSMNetwork, SSMNeuron

def test_neuron(duration=0.100, step_size=0.5e-3, input_current=100.0, tau_abs=0.03, tau_ref=0.010, u0=-0.065, r0=11.0, du=0.002, synapse_tau=0.010, rseed=123456, weight=0.01):

    np.random.seed(rseed)

    net = SSMNetwork(step_size=step_size)
    ssm = SSMNeuron()
    ssm.tau_abs = tau_abs
    ssm.tau_ref = tau_ref
    ssm.u0 = u0
    ssm.r0 = r0
    ssm.du = du
    ssm.synapse_tau = synapse_tau
    net.add(ssm)

    def ifunc(t):
        if t > 0.020:
            return input_current
        return 0.0
    ics = PoissonCurrentStream(ifunc, step_size)
    net.add_stream(ics)
    net.connect_stream(ics.id, 0, ssm.id, weight)
    net.compile()

    nsteps = int(duration / step_size)
    t = np.arange(nsteps)*step_size

    stime = time.time()
    state = np.zeros([nsteps, 1, 5])
    for k,ti in enumerate(t):
        state[k, 0, :] = net.state
        net.step()
    etime = time.time() - stime
    print 'Elapsed Time: %0.6fs' % etime

    st_index = state[:, 0, 4] > 0.0

    umin = -0.085
    umax = 0.050
    plt.figure()
    plt.subplot(4, 1, 1)
    u = state[:, 0, 0]
    u[st_index] = umax
    plt.plot(t, u, 'k-')
    plt.title('u(t)')
    plt.axis('tight')
    plt.ylim(umin, umax)
    plt.subplot(4, 1, 2)
    plt.plot(t, state[:, 0, 1], 'g-')
    plt.title('g(t)')
    plt.axis('tight')
    plt.subplot(4, 1, 3)
    plt.plot(t, state[:, 0, 2], 'r-')
    plt.title('R(t)')
    plt.axis('tight')
    plt.subplot(4, 1, 4)
    p = state[:, 0, 3]
    plt.plot(t, p, 'b-')
    for st in t[st_index]:
        plt.axvline(x=st, ymin=0.0, ymax=1.0, color='k', alpha=0.5)
    plt.axis('tight')
    plt.ylim(0.0, 1.0)
    plt.title('p(t)')




def plot_fi(duration=0.500, step_size=0.5e-3, tau_abs=0.03, tau_ref=0.010, u0=-0.065, r0=11.0, du=0.002, synapse_tau=0.010, rseed=123456, nreps=10, num_inputs=100):

    np.random.seed(rseed)
    #input_rates = np.arange(0.0, 150.0, 10.0)
    input_rates = [0.0, 25.0, 50.0, 75.0, 100.0, 150.0]

    net = SSMNetwork(step_size=step_size)
    ssm = SSMNeuron()
    ssm.tau_abs = tau_abs
    ssm.tau_ref = tau_ref
    ssm.u0 = u0
    ssm.r0 = r0
    ssm.du = du
    ssm.synapse_tau = synapse_tau
    net.add(ssm)

    num_spikes = []
    nsteps = int(duration / step_size)
    t = np.arange(nsteps)*step_size
    for irate in input_rates:

        def ifunc(t):
            if t > 0.020:
                return irate
            return 0.0
        for n in range(num_inputs):
            ics = PoissonCurrentStream(ifunc, step_size)
            net.add_stream(ics)
            net.connect_stream(ics.id, 0, ssm.id, 0.5)
        net.compile()

        nspikes = []
        for n in range(nreps):
            netcpy = copy.deepcopy(net)
            ns = 0
            for k,ti in enumerate(t):
                ns += netcpy.state[ssm.id, -1]
                netcpy.step()
            nspikes.append(ns)
        nspikes = np.array(nspikes)
        num_spikes.append(nspikes.mean())

    num_spikes = np.array(num_spikes)
    firing_rate = num_spikes / duration
    plt.figure()
    plt.plot(input_rates, firing_rate, 'ko-')
    plt.title('F/I Curve (%d neurons)' % num_inputs)
    plt.xlabel('Input Rate (Hz)')
    plt.ylabel('Firing Rate (Hz)')

