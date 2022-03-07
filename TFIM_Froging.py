import netket as nk
import numpy as np

from Forging_helper import *
from Forging_helper_Circuits_TFIM import *
import pennylane as qml
import os
from functools import partial

from netket import jax as nkjax
import jax
import jax.numpy as jnp
from jax import random
import time

import matplotlib.pyplot as plt

import pandas as pd

import optax
from netket.optimizer import Sgd, Adam


grid_size = [1, 4] # define grid size of TFIM model
n_qubits = np.prod(grid_size)
n_layers = 10
epochs = 5001
N = n_qubits //2

params_shape = (n_layers, N, 3)
alpha_beta_list = jnp.array([[0,0], [1,0],[0,1], [1,1]])
aj_list = jnp.array([1., -1., -1., 1.])
a0=1. 

shots=1000 # measurement shots of p(y|x) circuit
NN_features = 10 # intermediate dimension of NN model
NN_layers = 6 # Number of layers of NN model
random_H = False # random couplings for TFIM
seed = 41 # seed for rundom couplings of TFIM
lr = 0.001 #learning rate


# Get TFIM Hamiltonian and all required operators
Obs_AB, ha, H_full, J_overlap, J_subsys, h_subsys = get_all_operators_random_old(grid_size=grid_size, seed=seed, random=random_H)

Obs_AB = Obs_AB[::2] # <Z1ZN> = <ZNZ1>, therefore we only need to measure it once
J_AB = J_overlap
H_A = sum(ha).to_dense()
H = sum(H_full).to_dense()

number_of_overlapping_Ops = len(Obs_AB)
Hamiltonian_A = qml.Hermitian(H_A, wires=range(N))

Observables = []
pyx_indices = []

for Obs in Obs_AB:
    Observables.append(Obs[0])
    pyx_indices.append(list(Obs[0].wires))

# Get Classical Correlators <ZiZj>
e, v = np.linalg.eigh(H)
print("Min Energy: ", e[0])
ψ = v[:,0]
Classical_Correlators = calculate_all_correlators_classical(n_qubits, ψ)


# Initialize parameters and NN
key = random.PRNGKey(1)
key, subkey = random.split(key)
params = random.uniform(subkey, params_shape, dtype = np.float32)

hi = nk.hilbert.Spin(s=0.5, N=int(N))
sa = nk.sampler.ARDirectSampler(hi) # Sampler
model = nk.models.ARNNDense(hilbert=hi, layers= NN_layers, features=NN_features, dtype = np.float32)

s = jnp.ones(shape = (100, N))
_, subkey = random.split(subkey)
NN_params = model.init(subkey, s)

# Define all the circuits for given number of qubits
get_sample = partial(sample_NN, sa = sa, NN_model = model, n_qubits = n_qubits)
Circuits_subsys_and_diagonal = partial(Circuits_Observable, n_qubits = n_qubits)
qnode_y_to_state = partial(qnode_y_to_psi, n_qubits = n_qubits)
qnode_YX = partial(qnode_Y_given_X, n_qubits = n_qubits)
qnode_YX_states = partial(qnode_Y_given_X_states, n_qubits = n_qubits)

# Vmap all functions to get the right input batch dimensions
vmap_qnode_subsys = jax.vmap(Circuits_subsys_and_diagonal, in_axes=(None, 0, None), out_axes=0)
apply_fun_double_vmap = jax.vmap(jax.vmap(model.apply, in_axes=(None, 0), out_axes=0), in_axes=(None, 0), out_axes=0) 

triple_vmap_qnode_y_to_state = jax.jit(jax.vmap(jax.vmap(jax.vmap(qnode_y_to_state, in_axes=(0)), in_axes=(0)), in_axes=(0)))

# Create list of all qnode_YX circuits with corresponding indices.
# To jit them later we treat each circuit that measures different indices (ZiZj)
# as a new circuit. Otherwise we will get tracer errors.
vmap_list_sample = []
vmap_list_state = []
for i in range(number_of_overlapping_Ops):
    vmap_list_sample.append(jax.vmap(jax.vmap(partial(qnode_YX, indices = pyx_indices[i]), in_axes=(None, 0, 0, None), out_axes=0), in_axes=(None, None, None, 0), out_axes=0))
    vmap_list_state.append(jax.vmap(jax.vmap(partial(qnode_YX_states, indices = pyx_indices[i]), in_axes=(None, 0, 0, None), out_axes=0), in_axes=(None, None, None, 0), out_axes=0))
 

# Define all functions to sample expectation values
@jax.jit
def expect_value_OABs(params, Sample):
    """
    Get Expectation value for all observables in Observables list.
    These are the expectation <O_A*O_B> values of the operators acting on both 
    subsytems evaluated on one subsytem (first term eq.4 in paper).
    """
    μ_I_diagonal = 0.
    for j in range(number_of_overlapping_Ops):
        μ_I_diagonal += vmap_qnode_subsys(params, Sample, Observables[j])*a0*J_AB[j]
    return μ_I_diagonal

def body_sampling_no_grad(params, NN_params, Sample, key, f_sample):
    """
    Returns 2nd term of eq 4 in paper. For a given sample of the NN
    and all alpha, beta we get samples of p(y|x). We avaluate λ_{σ} for
    x and y.
    params: Parameters of VQE
    NN_params: Parameters of NN
    Sample: Sample of NN model
    key: PRNG key for jax
    f_sample: qnode_YX pennylane function for a fixed index i,j.
    """
    key, subkey = random.split(key)
    subkeys = jax.random.split(subkey, len(Sample))

    Y = jax.lax.stop_gradient(f_sample(params, Sample, subkeys, alpha_beta_list))
    Y_new = (Y-0.5)*2
    psi_y = jnp.exp(apply_fun_double_vmap(NN_params, Y_new)) # We once had here a -1*Y, because of old pennylane convention
    psi_x = jnp.exp(model.apply(NN_params, 2*(Sample-0.5)))
    factor = (psi_y / psi_x.reshape(1,-1,1)) # **2 because model.apply gives sqrt(p)
    μ_XY = (aj_list.reshape(-1,1,1)*(factor)).mean(axis=(0,2))/2
    return jnp.real(μ_XY)


def body_sampling_with_grad(params, NN_params, Sample, key, f_sample, f_state):
    """
    Same function as the "no grad" version, but now we can use it for the backpass step in equation 12, where we evaluate the gradients wrt the VQE parameters. This step is not scalable. For big systems, we need gradient free optimzation. Such as SPSA. 
    params: Parameters of VQE
    NN_params: Parameters of NN
    Sample: Sample of NN model
    key: PRNG key for jax
    f_sample: qnode_YX pennylane qnode for a fixed index i,j that returns samples.
    f_state: qnode_YX_state pennylane function for a fixed index i,j that returns state vectors.
    """
    key, subkey = random.split(key)
    subkeys = jax.random.split(subkey, len(Sample))
    Y = jax.lax.stop_gradient(f_sample(params, Sample, subkeys, alpha_beta_list))
    Y_new = (Y - 0.5)*2 # go from {-1, +1} to {0,1}
    states_Y = jax.lax.stop_gradient(triple_vmap_qnode_y_to_state(Y))

    states_UCUx = f_state(params, Sample, subkeys, alpha_beta_list)
    shape = states_UCUx.shape

    F = jnp.sum(states_Y*states_UCUx.reshape(shape[0], shape[1], 1, shape[2]), axis = 3)
    p_xy = jnp.conjugate(F)*F
    all_probs = jnp.log(p_xy)

    psi_y = jnp.exp(apply_fun_double_vmap(NN_params, Y_new))
    psi_x = jnp.exp(model.apply(NN_params, 2*(Sample-0.5)))
    factor = (psi_y / psi_x.reshape(1,-1,1))
    μ_XY = 0.5*(aj_list.reshape(-1,1,1)*((factor * all_probs) - all_probs.mean()*factor.mean())).mean(axis=(0,2))
    return jnp.real(μ_XY)


@jax.jit
def sampling_psiy(params, NN_params, Sample, key):
    """Evaluates 2nd term of eq 4 in paper for all overlapping terms"""
    μ_XY = 0.
    for j in range(number_of_overlapping_Ops):
        μ_XY += body_sampling_no_grad(params, NN_params, Sample, key, vmap_list_sample[j])*J_AB[j]
    return jnp.real(μ_XY)

@jax.jit
def sampling_with_probs(params, NN_params, Sample, key):
    """Evaluates 2nd term of eq 4 in paper for all overlapping terms with probs for gradient descent"""
    μ_XY = 0.
    for j in range(number_of_overlapping_Ops):
        μ_XY += body_sampling_with_grad(params, NN_params, Sample, key, vmap_list_sample[j], vmap_list_state[j])*J_AB[j]
    return jnp.real(μ_XY)

@partial(jax.jit, static_argnums=4)
def expect_value(params, NN_params, Sample, key, Hamiltonian_A):
    """Evaluate energy expectation value of all terms combined"""
    μ_HA = vmap_qnode_subsys(params, Sample, Hamiltonian_A)
    μ_XY = sampling_psiy(params, NN_params, Sample, key)
    μ_OAB = expect_value_OABs(params, Sample)
    return 2*(μ_HA + μ_OAB + μ_XY) 
    # factor 2 because we have two subsystems H_A and the couplings O_AB are counted twice

def loss_NN_new(params, NN_params, Sample, key):
    """loss function to evaluate gradient wrt NN parameters"""
    μ_HA = vmap_qnode_subsys(params, Sample, Hamiltonian_A)
    μ_OAB = expect_value_OABs(params, Sample)
    μ_XY = sampling_psiy(params, NN_params, Sample, key)
    μ_XY_no_grad = jax.lax.stop_gradient(μ_XY)
    log_p = 2*model.apply(NN_params, (Sample - 0.5)*2.) #factor 2 because model.apply returns logψ and not log ψ^2
    measure = 2*(μ_HA + μ_OAB + μ_XY_no_grad)
    return (log_p*measure - log_p.mean()*measure.mean() + 2*μ_XY).mean()

def loss_U_new(params, NN_params, Sample, key):
    """loss function to evaluate gradient wrt VQE parameters"""
    μ_HA = vmap_qnode_subsys(params, Sample, Hamiltonian_A)
    μ_OAB = expect_value_OABs(params, Sample)
    μ_XY = sampling_with_probs(params, NN_params, Sample, key)
    return (2*(μ_HA + μ_OAB + μ_XY)).mean()

# Jit all the gradients
s, S = get_sample(NN_params, chain_length=128)
start = time.time()
grad_U_fn = jax.jit(nkjax.value_and_grad(loss_U_new, argnums = 0))
lU, gU = grad_U_fn(params, NN_params, S, key)
end1 = time.time()
grad_NN_fn = jax.jit(nkjax.value_and_grad(loss_NN_new, argnums = 1))
lNN, gNN = grad_NN_fn(params, NN_params, S, key)
end2 = time.time()

jit_time1 = (end1-start)/60
jit_time2 = (end2-start)/60

print("Time to Jit U grads: ", jit_time1, " Minutes")
print("Time to Jit NN grads: ", jit_time2, " Minutes")

# Reinitialize all the parameters
_, subkey = random.split(subkey)
NN_params = model.init(subkey, s)
params = random.uniform(subkey, params_shape, dtype = np.float32)
# E = expect_value(params, NN_params, S, key, Hamiltonian_A)

# s, S = get_sample(NN_params)


optU = Adam(learning_rate=lr)
optNN = Adam(learning_rate=lr)
opt_stateU = optU.init(params)
opt_stateNN = optNN.init(NN_params)
opt = Adam(learning_rate=lr)
opt_state = opt.init((params, NN_params))

measure_progress = []
params_progress = []
NN_progress = []

for i in range(epochs):
    _, subkey = random.split(subkey)
    s, S = get_sample(NN_params, chain_length=512)
    
    if i%10 == 0:
        M = expect_value(params, NN_params, S, subkey, Hamiltonian_A).mean()
        measure_progress.append(M)
        params_progress.append(params)
        NN_progress.append(NN_params)
        print('Loss step {}: '.format(i), M)

    if i < 1000:
        measure, grads_U = grad_U_fn(params, NN_params, S, subkey)
        updatesU, opt_stateU = optU.update(grads_U, opt_stateU)
        params = optax.apply_updates(params, updatesU)
    else:
        if i%10 != 0:
            measure, grads_U = grad_U_fn(params, NN_params, S, subkey)
            updatesU, opt_stateU = optU.update(grads_U, opt_stateU)
            params = optax.apply_updates(params, updatesU)
        else:
            loss_val, grads_NN = grad_NN_fn(params, NN_params, S, subkey)
            updatesNN, opt_stateNN = optNN.update(grads_NN, opt_stateNN)
            NN_params = optax.apply_updates(NN_params, updatesNN)
    
    if i%100 == 0:
        df = pd.DataFrame()
        new_row = pd.Series(data={"Measure": measure_progress, "U_params": params_progress, "NN_params": 
                            NN_progress, "MinEnergy":e[0],
                            "n_qubits": n_qubits, "shots": shots, "n_layers": n_layers, 
                            "NN_layers": NN_layers, "NN_features": NN_features,
                            "epochs": epochs, "lr": lr, "J_AB": J_AB, "J_subsys": J_subsys, "h_subsys":h_subsys, "min_energy":e[0]}, 
                            name='{}'.format(0))
        df = df.append(new_row, ignore_index= False)
    


df = pd.DataFrame()
new_row = pd.Series(data={"Measure": measure_progress, "U_params": params_progress, "NN_params": 
                     NN_progress, "MinEnergy":e[0],
                    "n_qubits": n_qubits, "shots": shots, "n_layers": n_layers, 
                    "NN_layers": NN_layers, "NN_features": NN_features,
                    "epochs": epochs, "lr": lr, "J_AB": J_AB, "J_subsys": J_subsys, "h_subsys":h_subsys, "min_energy":e[0]}, 
                    name='{}'.format(0))
df = df.append(new_row, ignore_index= False)

    

PATH = "Forging_Data/Forging_grid_{}_trial_{}.pkl".format(grid_size, trial)
df.to_pickle(PATH)


## Calculate all correlators ZiZj
def correlation(OP_1, OP_2, idx1, idx2, n_qubits, params = None, NN_params = None, V= qml.Identity(0), a0=1., aj=[1.,-1.,-1.,1.]):    
    pyx_indices = [[idx1%N, idx2%N]]
    number_of_overlapping_Ops = 1

    n_qubits = n_qubits

    if idx1%N != idx2%N:
        Observable = OP_1(idx1%N) @ OP_2(idx2%N) 
    else:
        Observable = qml.Identity(0)

    Obs_Correlator = Observable # Can be generalized later to many Obs

    if (idx1 < N and idx2 < N) or (idx1 >= N and idx2 >= N):
        # Both in subsystem A
        print(idx1, idx2, "same system")
        both_subsystems = False

    else:
        # One in A, the other in B
        print(idx1, idx2, "other system")
        both_subsystems = True
        

    vmap_qnode_Correlator = jax.vmap(jax.vmap(partial(qnode_YX, indices = [idx1%N, idx2%N]), in_axes=(None, 0, 0, None), out_axes=0), in_axes=(None, None, None, 0), out_axes=0)

    def XY_Correlator(params, NN_params, Sample, key):
        μ_XY = body_sampling_no_grad(params, NN_params, Sample, key, vmap_qnode_Correlator)
        return jnp.real(μ_XY)

    def OAB_Correlator(params, NN_params, Sample, key):
        μ_I_diagonal = vmap_qnode_subsys(params, Sample, Obs_Correlator)
        return μ_I_diagonal


    def expect_value_Correlator(params, NN_params, Sample, key, both_subsys = True):
        if both_subsys:
            μ_XY = XY_Correlator(params, NN_params, Sample, key)
            μ_I_diagonal = OAB_Correlator(params, NN_params, Sample, key)
        else:
            μ_I_diagonal = OAB_Correlator(params, NN_params, Sample, key)
            μ_XY = 0.
        return μ_I_diagonal + jnp.real(μ_XY)
    
    s, S = get_sample(NN_params, chain_length = 1000)
    return expect_value_Correlator(params, NN_params, S, subkey).mean()
        
def calculate_all_correlators_quantum(n_qubits, OP_1 = qml.PauliZ, OP_2 = qml.PauliZ, 
                                      params = None, NN_params = NN_params, key = random.PRNGKey(1), grid_size = [1,4]):
    """
    n_qubits: are here number of qubits of composite system
    """
    
    for idx1 in range(n_qubits):
        for idx2 in range(n_qubits):
            if idx1 < idx2:    
                correlation_dict = {}
                for idx1 in range(n_qubits):
                    for idx2 in range(n_qubits):
                        if idx1 < idx2:
                            ZZ = correlation(OP_1, OP_2, idx1, idx2, n_qubits, V= qml.Identity(0), 
                                             a0=1., aj=[1.,-1.,-1.,1.], params = params, 
                                             NN_params = NN_params)
                            correlation_dict["{}{}".format(idx1, idx2)] = ZZ.mean().item()
                return correlation_dict


Quantum_Correlators = calculate_all_correlators_quantum(n_qubits, params= params, NN_params = NN_params)
Classical_Correlators = calculate_all_correlators_classical(n_qubits, ψ)




plt.figure(figsize=(10,6))
plt.bar(Quantum_Correlators.keys(), Quantum_Correlators.values(), 0.7, color='b', alpha = 0.3)
plt.bar(Classical_Correlators.keys(), Classical_Correlators.values(), 0.7, color='g', alpha = 0.3)


C_path = "Forging_Data/Correlator_quantum_lr_{}_random_{}_grid_{}_trial_{}".format(lr, random_H, grid_size, trial)
np.save(C_path, Quantum_Correlators)

C_path = "Forging_Data/Correlator_classical_lr_{}_random_{}_grid_{}_trial_{}".format(lr, random_H, grid_size, trial)
np.save(C_path, Classical_Correlators)



