#!/usr/bin/env python
# coding: utf-8
import netket as nk
from netket.operator.spin import sigmax,sigmaz
from netket import jax as nkjax
import jax
import jax.numpy as jnp
from functools import partial
# from jax import random
# from tqdm import tqdm
import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from pennylane.operation import AnyWires, Operation
from pennylane.wires import Wires

from Forging_helper import *

def Cj_template(alpha, beta, idx1, idx2):
    """
    For the training we use the U3 to implement X^alpha.
    The problem with this function is that it doesn't work properly for idx1 == idx2
    This does not cause problems for TFIM with nearest neighbour interactions
    But it can cause problems for the fermionic Hamiltonians.
    """
    qml.U3(np.pi*alpha, 0., np.pi*alpha, wires = idx1) # We use U3 because RX gives a phase
    qml.U3(np.pi*beta, 0., np.pi*beta, wires = idx2)
    if not idx1 == idx2:
        qml.CZ(wires = [idx1, idx2])
    else:
        qml.U1(((alpha-0.5)*2. + (beta-0.5)*2.)*np.pi/2, wires=idx1)
    qml.U3(np.pi*alpha, 0., np.pi*alpha, wires = idx1)
    qml.U3(np.pi*beta, 0., np.pi*beta, wires = idx2)

def Cj_template_general(alpha, beta, O1 = qml.PauliX(0), O2 = qml.PauliX(1)):
    """
    Cj template for arbitrary O1 and O2
    """
    o1o2 = jnp.matmul(O1, O2) 
    one = jnp.eye(O1.shape[0])
    M = 0.5*(one + (-1.)**alpha*O1 + (-1.)**beta*O2 - (-1.)**(alpha + beta)*o1o2)
    return M

def sample_NN(NN_params, chain_length = 128, sa = None, NN_model = None, n_qubits = 4):
    """
    get sample from NN in $s \in {-1, 1}$ and $S \in {0,1}$ conversion.
    NN_params: Parameters of classical model
    chain_length: Number of samples
    sa: Netket sampler
    NN_model: Netket NN model
    """
    Sample, _ = nk.sampler.ARDirectSampler.sample(sa, NN_model, NN_params, chain_length = chain_length)
    # Sample, _ = nk.sampler.sample(sa, NN_model, NN_params, chain_length = chain_length)
    Sample = Sample.reshape(-1, n_qubits//2)
    s = jax.lax.stop_gradient(Sample)
    S = (s + 1)/2
    S = S.astype(int)
    return s, S

def Circuits_Observable(params, inputs, Observable, n_qubits = 2):
    """
    params: Parameters of the VQE
    inputs: Samples of the NN
    Observable: Is either the Hamiltonian of the subsystem, if we calculate <H_A>
    or it is the operator O_1 O_2 (as a multiplication not a tensor product) as in
    eq 11 (https://arxiv.org/pdf/2104.10220.pdf)
    n_qubits: refers here to number of qubits of whole system
    """
    dev = qml.device('default.qubit.jax', wires=n_qubits//2)    
    @partial(jax.jit, static_argnums=2)    
    @qml.qnode(dev, interface='jax', diff_method="backprop")
    def qnode(params, inputs, Observable):
        for i in range(n_qubits//2):
            qml.RX(jnp.pi*inputs[i], wires=i)
        brick_wall_entangling(params)
        return qml.expval(Observable)  
    return qnode(params, inputs, Observable) 

def qnode_y_to_psi(inputs, n_qubits):
    """
    Translate a sample into a state vector with pennylane conversion
    N: Is nr of qubits of subsystem!!
    """
    dev = qml.device('default.qubit.jax', wires=n_qubits//2, shots = None)
    @jax.jit
    @qml.qnode(dev, interface='jax')
    def circuit(inputs):
        for i in range(n_qubits//2):
            qml.RX(jnp.pi*inputs[i], wires=i)
        return qml.state()

    return circuit(inputs)

def qnode_Y_given_X(params, inputs, key, alpha_beta, indices, n_qubits = 2):
    """
    Circuit to sample p(y|x) for operators ZiZj that act on different subsytems.
    params: Parameters of VQE
    inputs: Samples from NN model
    key: PRNG key for jax
    alpha_beta: list of tuple with [alpha, beta]
    indices: which index the operator Zi Zj is acting on., where i and j are mod(N)
    with N = n_qubits//2
    n_qubits: Is nr of qubits of full system!!
    """
    idx1, idx2 = indices
    alpha, beta = alpha_beta
    U = partial(Cj_template, alpha, beta, idx1 = idx1, idx2 = idx2)
    dev = qml.device('default.qubit.jax', wires=n_qubits//2, shots = 1000, prng_key = key)

    @partial(jax.jit, static_argnums = 2)
    @qml.qnode(dev, interface='jax', diff_method=None)
    def circuit(params, inputs, U):
        for i in range(n_qubits//2):
            qml.RX(jnp.pi*inputs[i], wires=i)
        brick_wall_entangling(params)
        U()
        qml.adjoint(brick_wall_entangling)(params)
        return qml.sample()

    return circuit(params, inputs, U)

def qnode_Y_given_X_states(params, inputs, key, alpha_beta, indices, n_qubits = 2):
    """
    Same Circuit as to sample p(y|x) for operators ZiZj that act on different subsytems.
    But now it returns the state vector.
    params: Parameters of VQE
    inputs: Samples from NN model
    key: PRNG key for jax
    alpha_beta: list of tuple with [alpha, beta]
    indices: which index the operator Zi Zj is acting on., where i and j are mod(N)
    with N = n_qubits//2
    n_qubits: Is nr of qubits of full system!!
    """
    dev = qml.device('default.qubit.jax', wires=n_qubits//2)
    idx1, idx2 = indices
    alpha, beta = alpha_beta
    U = partial(Cj_template, alpha, beta, idx1 = idx1, idx2 = idx2)   
    @partial(jax.jit, static_argnums = 2)
    @qml.qnode(dev, interface='jax', diff_method="backprop")
    def circuit(params, inputs, U):
        for i in range(n_qubits//2):
            qml.RX(jnp.pi*inputs[i], wires=i)
        brick_wall_entangling(params)
        U()
        qml.adjoint(brick_wall_entangling)(params)
        return qml.state()
    return circuit(params, inputs, U)

def qnode_Y_given_X_fermion(params, inputs, key, alpha_beta, Op_AB, n_qubits):
    """
    n_qubits: Is nr of qubits of full system!!
    """
    alpha, beta = alpha_beta
    m = Cj_template_general(alpha=alpha, beta = beta, O1 = jnp.array(Op_AB[1].matrix), O2 = jnp.array(Op_AB[2].matrix)) # index 0 is the prefactor
    U = partial(apply_unitary, M=m, wires = range(n_qubits//2))
    dev = qml.device('default.qubit.jax', wires=n_qubits//2, shots = 1000, prng_key = key)

    @partial(jax.jit, static_argnums = 2)
    @qml.qnode(dev, interface='jax', diff_method=None)
    def circuit(params, inputs, U):
        for i in range(n_qubits//2):
            qml.RX(jnp.pi*inputs[i], wires=i)
        brick_wall_entangling(params)
        U()
        qml.adjoint(brick_wall_entangling)(params)
        return qml.sample()

    return circuit(params, inputs, U)



def qnode_Y_given_X_states_fermion(params, inputs, key, alpha_beta, Op_AB, n_qubits):
    """
    n_qubits: Is nr of qubits of full system!!
    """
    dev = qml.device('default.qubit.jax', wires=n_qubits//2)
    
    alpha, beta = alpha_beta
    m = Cj_template_general(alpha=alpha, beta = beta, O1 = jnp.array(Op_AB[1].matrix), O2 = jnp.array(Op_AB[2].matrix)) # index 0 is the prefactor
    U = partial(apply_unitary, M=m, wires = range(n_qubits//2))
    
    @partial(jax.jit, static_argnums = 2)
    @qml.qnode(dev, interface='jax', diff_method="backprop")
    def circuit(params, inputs, U):
        for i in range(n_qubits//2):
            qml.RX(jnp.pi*inputs[i], wires=i)
        brick_wall_entangling(params)
        U()
        qml.adjoint(brick_wall_entangling)(params)
        return qml.state()

    return circuit(params, inputs, U)