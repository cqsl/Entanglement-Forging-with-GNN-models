from audioop import mul
import netket as nk
# from netket.operator.spin import sigmax,sigmaz

# from Forging_helper_updated import *
from Forging_helper import *
from Forging_helper_Circuits_TFIM import *
import os
from functools import partial

from netket import jax as nkjax
import jax
import jax.numpy as jnp
from jax import random

from tqdm import tqdm
import time

import matplotlib.pyplot as plt

import pandas as pd

import optax
from netket.optimizer import Sgd, Adam

import pennylane as qml
import numpy as np
import openfermion as of
from openfermion import jordan_wigner
from openfermion import get_sparse_operator, get_ground_state


n_layers = 8
epochs = 5001
alpha_beta_list = jnp.array([[0,0], [1,0],[0,1], [1,1]])

# aj_list for spinless fermions 
# How to determine the prefactors aj and a0
# This was done manually. Check https://arxiv.org/pdf/2104.10220.pdf SM2
# YZ, YI, sigma = -1, commute, aj = [-1,1,1,-1], a0 = -1
# XZ, XI, sigma = 1, commute, aj = [1,-1,-1,1], a0 = 1
# ZI, ZI, sigma = 1, O1 = O2, aj = [1,1,0,0], a0 = 0
# IY, ZY, sigma = -1, commute, aj = [-1,1,1,-1], a0 = -1
# IX, ZX, sigma = 1, commute, aj = [1,-1,-1,1], a0 = 1
# IZ, IZ, sigma = 1, O1 = O2, aj = [1,1,0,0]


ajs = [[-1.0, 1.0, 1.0, -1.0], [1.0, -1.0, -1.0, 1.0], [1., 1., 0., 0.]]*2  
aj_list = jnp.array(ajs)
shots=1000 
a0 = jnp.array([-1., 1., 0., -1., 1., 0.]) # also a0 depends on if O1,O2 commute or are equal or anti-commute
# aj=[1.,-1.,-1.,1.] 
NN_Samples = 100
NN_features = 10
NN_layers = 4

lr = 0.001

"""Define the Hamiltonian."""
# Parameters.
nsites = 2
U = 1.0
J = 1.0
spinless = True

hubbard = of.fermi_hubbard(2, nsites, tunneling=-J, coulomb=U, periodic=True, spinless = spinless, particle_hole_symmetry=True)
h1 = jordan_wigner(hubbard)

H = get_sparse_operator(hubbard)
E = get_ground_state(H)[0]
print("GS energy:", get_ground_state(H)[0])


# ## Get all oprators for forging
# 
# The output of this are lists with operators acting on A and acting on B and their respsective prefactor
local_operators_A = []
local_operators_B = []
overlap_operators = []
gates = {"X": qml.PauliX, "Y": qml.PauliY, "Z": qml.PauliZ, "I":qml.Identity}

n_qubits = 0
for key in h1.terms.keys():
    for k in key:
        if k[0] > n_qubits:
            n_qubits = k[0]

n_qubits += 1        
N = n_qubits//2
print("number_of_qubits: ", n_qubits)
print("Subsystem size: ", N)   

for o in h1:
    for keys in o.terms.keys():
        indices = np.array(keys)[:,0].astype(np.int64)
        Ops = np.array(keys)[:,1]
#         print(indices)
        op_list = [[np.real(x) for x in o.terms.values()][0]] # first part of op_list is pre factor
        for i in range(n_qubits//2):
            if np.isin(i, indices):
                idx = np.where(indices == i)
                O = gates[Ops[idx][0]](i)
            else:
                O = gates["I"](i)
            if i == 0:
                op = O
            else:
                op = op@O
        op_list.append(op)
        for i in range(n_qubits//2, n_qubits):
            if np.isin(i, indices):
                idx = np.where(indices == i)
                O = gates[Ops[idx][0]](i%(n_qubits//N))
            else:
                O = gates["I"](i%(n_qubits//2))
            if i == n_qubits//2:
                op = O
            else:
                op = op@O
        op_list.append(op)
        if ((indices < N)*1.0).mean() == 1: # All indices in A
            local_operators_A.append(op_list)
        elif ((indices < N)*1.0).mean() == 0.: # All indices in B
            local_operators_B.append(op_list)
        else:
            overlap_operators.append(op_list)


H_A = 0
for h in local_operators_A:
    H_A += h[0]*h[1].matrix
    
H_A

Hamiltonian_A = qml.Hermitian(H_A, wires = range(n_qubits//2))


N = n_qubits //2
params_shape = (n_layers, N, 3)



Obs_AB = overlap_operators
number_of_overlapping_Ops = len(Obs_AB)



key = random.PRNGKey(1)
key, subkey = random.split(key)
params = random.uniform(subkey, params_shape, dtype = np.float32)


            

hi = nk.hilbert.Spin(s=0.5, N=int(N))
sa = nk.sampler.ARDirectSampler(hi) # Sampler
model = nk.models.ARNNDense(hilbert=hi, layers= NN_layers, features=NN_features, dtype = np.float32)

s = jnp.ones(shape = (100, N))
_, subkey = random.split(subkey)
NN_params = model.init(subkey, s)

get_sample = partial(sample_NN, sa = sa, NN_model = model, n_qubits = n_qubits)

Circuits_subsys_and_diagonal = partial(Circuits_Observable, n_qubits = n_qubits)
qnode_y_to_state = partial(qnode_y_to_psi, n_qubits = n_qubits)
qnode_YX = partial(qnode_Y_given_X_fermion, n_qubits = n_qubits)
qnode_YX_states = partial(qnode_Y_given_X_states_fermion, n_qubits = n_qubits)
# def get_sample(NN_params, chain_length = 128):
#     Sample, _ = nk.sampler.sample(sa, model, NN_params, chain_length = chain_length)
#     Sample = Sample.reshape(-1, N)
#     s = jax.lax.stop_gradient(Sample)
#     S = (s + 1)/2
#     S = S.astype(int)
#     return s, S







# def Circuits_subsys_and_diagonal(params, inputs, Observable, n_qubits):
#     """
#     Observable: Is either the Hamiltonian of the subsystem, if we calculate <H_A>
#     or it is the operator O_1 O_2 (as a multiplication not a tensor product) as in
#     eq 11 (https://arxiv.org/pdf/2104.10220.pdf)
#     n_qubits: refers here to number of qubits in subsystem
#     """
#     dev = qml.device('default.qubit.jax', wires=n_qubits//2)
# #     qml.Hermitian(H_A, wires=range(n_qubits//2))
      
#     @qml.qnode(dev, interface='jax', diff_method="backprop")
#     def qnode(params, inputs, Observable):
#         for i in range(n_qubits//2):
#             qml.RX(jnp.pi*inputs[i], wires=i)
#         brick_wall_entangling(params)
#         return qml.expval(Observable)
    
#     return qnode(params, inputs, Observable)  




# def qnode_y_to_state(inputs):
#     """
#     Translate a sample into a state vector with pennylane conversion
#     N: Is nr of qubits of subsystem!!
#     """
#     dev = qml.device('default.qubit.jax', wires=N, shots = None)
    
#     @jax.jit
#     @qml.qnode(dev, interface='jax')
#     def circuit(inputs):
#         for i in range(N):
#             qml.RX(jnp.pi*inputs[i], wires=i)
#         return qml.state()

#     return circuit(inputs)



vmap_qnode_subsys = jax.vmap(Circuits_subsys_and_diagonal, in_axes=(None, 0, None), out_axes=0) 
               
def expect_value_OABs(params, NN_params, Sample, key):
    """ 
    There is a bug in pennylane 0.21, that prevents us from simply using 
    Obs_AB[j][1]@Obs_AB[j][2]. This is why we do it like this
    """
    μ_I_diagonal = 0.
    for j in range(number_of_overlapping_Ops):
        Observable = qml.Hermitian(np.matmul(Obs_AB[j][1].matrix, Obs_AB[j][2].matrix), wires = Obs_AB[j][1].wires)
        μ_I_diagonal += vmap_qnode_subsys(params, Sample, Observable)*a0[j]*Obs_AB[j][0] # idx 0 is prefactor, idx 1 is operator
    return μ_I_diagonal


# from pennylane.operation import AnyWires, Operation
# from pennylane.wires import Wires

# class QubitUnitary(Operation):
#     num_wires = AnyWires
#     num_params = 1
#     grad_method = None

#     def __init__(self, *params, wires, do_queue=True):
#         wires = Wires(wires)

#         super().__init__(*params, wires=wires, do_queue=do_queue)

#     @classmethod
#     def _matrix(cls, *params):
#         return params[0]


#     def adjoint(self):
#         return QubitUnitary(qml.math.T(qml.math.conj(self.matrix)), wires=self.wires)


#     def label(self, decimals=None, base_label=None):
#         return super().label(decimals=decimals, base_label=base_label or "U")


# def apply_unitary(M, wires):
#     QubitUnitary(M, wires=wires)

# f = partial(apply_unitary, M, range(n_qubits//2))



# ## Can at the moment not be jitted

# In[568]:


# def qnode_Y_given_X_fermion(params, inputs, key, alpha_beta, Op_AB, n_qubits):
#     """
#     n_qubits: Is nr of qubits of full system!!
#     """
#     alpha, beta = alpha_beta
#     m = Cj_template_general(alpha=alpha, beta = beta, O1 = jnp.array(Op_AB[1].matrix), O2 = jnp.array(Op_AB[2].matrix)) # index 0 is the prefactor
#     U = partial(apply_unitary, M=m, wires = range(n_qubits//2))
#     dev = qml.device('default.qubit.jax', wires=n_qubits//2, shots = 1000, prng_key = key)

#     @partial(jax.jit, static_argnums = 2)
#     @qml.qnode(dev, interface='jax', diff_method=None)
#     def circuit(params, inputs, U):
#         for i in range(n_qubits//2):
#             qml.RX(jnp.pi*inputs[i], wires=i)
#         brick_wall_entangling(params)
#         U()
#         qml.adjoint(brick_wall_entangling)(params)
#         return qml.sample()

#     return circuit(params, inputs, U)



# def qnode_Y_given_X_states_fermion(params, inputs, key, alpha_beta, Op_AB, n_qubits):
#     """
#     n_qubits: Is nr of qubits of full system!!
#     """
#     dev = qml.device('default.qubit.jax', wires=n_qubits//2)
    
#     alpha, beta = alpha_beta
#     m = Cj_template_general(alpha=alpha, beta = beta, O1 = jnp.array(Op_AB[1].matrix), O2 = jnp.array(Op_AB[2].matrix)) # index 0 is the prefactor
#     U = partial(apply_unitary, M=m, wires = range(n_qubits//2))
    
#     @partial(jax.jit, static_argnums = 2)
#     @qml.qnode(dev, interface='jax', diff_method="backprop")
#     def circuit(params, inputs, U):
#         for i in range(n_qubits//2):
#             qml.RX(jnp.pi*inputs[i], wires=i)
#         brick_wall_entangling(params)
#         U()
#         qml.adjoint(brick_wall_entangling)(params)
#         return qml.state()

#     return circuit(params, inputs, U)




apply_fun_double_vmap = jax.vmap(jax.vmap(model.apply, in_axes=(None, 0), out_axes=0), in_axes=(None, 0), out_axes=0) 

triple_vmap_qnode_y_to_state = jax.jit(jax.vmap(jax.vmap(jax.vmap(qnode_y_to_state, in_axes=(0)), in_axes=(0)), in_axes=(0)))



vmap_list_sample = []
vmap_list_state = []
for i in range(number_of_overlapping_Ops):
    vmap_list_sample.append(jax.vmap(jax.vmap(partial(qnode_YX, Op_AB = Obs_AB[i]), in_axes=(None, 0, 0, None), out_axes=0), in_axes=(None, None, None, 0), out_axes=0))
    vmap_list_state.append(jax.vmap(jax.vmap(partial(qnode_YX_states, Op_AB = Obs_AB[i]), in_axes=(None, 0, 0, None), out_axes=0), in_axes=(None, None, None, 0), out_axes=0))


def body_sampling_no_grad(params, NN_params, Sample, key, f_sample, aj):
    key, subkey = random.split(key)
    subkeys = jax.random.split(subkey, len(Sample))

    Y = jax.lax.stop_gradient(f_sample(params, Sample, subkeys, alpha_beta_list))
    Y_new = (Y-0.5)*2
    psi_y = jnp.exp(apply_fun_double_vmap(NN_params, Y_new)) # We once had here a -1*Y, because of old pennylane convention
    psi_x = jnp.exp(model.apply(NN_params, 2*(Sample-0.5)))
    factor = (psi_y / psi_x.reshape(1,-1,1)) # **2 because model.apply gives sqrt(p)
    μ_XY = (aj.reshape(-1,1,1)*(factor)).mean(axis=(0,2))/2
    return jnp.real(μ_XY)


def body_sampling_with_grad(params, NN_params, Sample, key, f_sample, f_state, aj):
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
    μ_XY = 0.5*(aj.reshape(-1,1,1)*((factor * all_probs) - all_probs.mean()*factor.mean())).mean(axis=(0,2))
    return jnp.real(μ_XY)



@jax.jit
def sampling_psiy(params, NN_params, Sample, key):
    μ_XY = 0.
    for j in range(number_of_overlapping_Ops):
        μ_XY += body_sampling_no_grad(params, NN_params, Sample, key, vmap_list_sample[j], aj_list[j])*Obs_AB[j][0] #index 0 of Obs_AB are the prefactors
    return jnp.real(μ_XY)

@jax.jit
def sampling_with_probs(params, NN_params, Sample, key):
    μ_XY = 0.
    for j in range(number_of_overlapping_Ops):
        μ_XY += body_sampling_with_grad(params, NN_params, Sample, key, vmap_list_sample[j], vmap_list_state[j], aj_list[j])*Obs_AB[j][0]
    return jnp.real(μ_XY)



@partial(jax.jit, static_argnums=4)
def expect_value(params, NN_params, Sample, key, Hamiltonian_A):
    μ_HA = vmap_qnode_subsys(params, Sample, Hamiltonian_A)
    μ_XY = sampling_psiy(params, NN_params, Sample, key)
    μ_OAB = expect_value_OABs(params, NN_params, Sample, key)
    return 2*μ_HA + μ_OAB + μ_XY # onyl μ_HA with factor 2 for fermions



def loss_NN_new(params, NN_params, Sample, key):
    μ_HA = vmap_qnode_subsys(params, Sample, Hamiltonian_A)
    μ_OAB = expect_value_OABs(params, NN_params, Sample, key)
    μ_XY = sampling_psiy(params, NN_params, Sample, key)
    μ_XY_no_grad = jax.lax.stop_gradient(μ_XY)
    log_p = 2*model.apply(NN_params, (Sample - 0.5)*2.)
    measure = 2*μ_HA + μ_OAB + μ_XY_no_grad
    return (log_p*measure - log_p.mean()*measure.mean() + μ_XY).mean()

def loss_U_new(params, NN_params, Sample, key):
    μ_HA = vmap_qnode_subsys(params, Sample, Hamiltonian_A)
    μ_OAB = expect_value_OABs(params, NN_params, Sample, key)
    μ_XY = sampling_with_probs(params, NN_params, Sample, key)
    return (2*μ_HA + μ_OAB + μ_XY).mean()


s, S = get_sample(NN_params, chain_length = 128)
_, subkey = random.split(subkey)
NN_params = model.init(subkey, s)
params = random.uniform(subkey, params_shape, dtype = np.float32)
# E = expect_value(params, NN_params, S, key, Hamiltonian_A)



start = time.time()
print("start Jitting")
grad_U_fn = jax.jit(nkjax.value_and_grad(loss_U_new, argnums = 0))
lU, gU = grad_U_fn(params, NN_params, S, key)
print("U grads jitted")
end1 = time.time()
grad_NN_fn = jax.jit(nkjax.value_and_grad(loss_NN_new, argnums = 1))
lNN, gNN = grad_NN_fn(params, NN_params, S, key)
print("NN grads jitted")
end2 = time.time()

jit_time1 = (end1-start)/60
jit_time2 = (end2-start)/60

print("Time to Jit U grads: ", jit_time1, " Minutes")
print("Time to Jit NN grads: ", jit_time2, " Minutes")


_, subkey = random.split(subkey)
NN_params = model.init(subkey, s)
params = random.uniform(subkey, params_shape, dtype = np.float32)



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
    s, S = get_sample(NN_params, chain_length=1024)
    
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
    


# # In[ ]:


# df = pd.DataFrame()
# shots = 1000

# new_row = pd.Series(data={"Measure": measure_progress, "U_params": params_progress, "NN_params": NN_progress,
#                     "n_qubits": n_qubits, "shots": shots, "n_layers": n_layers, 
#                     "NN_Samples": NN_Samples, "NN_layers": NN_layers, "NN_features": NN_features,
#                     "epochs": epochs, "lr": lr, "is_complex" : complex_true, "min_energy":E, "U":U, "J":J, "Spinless":spinless}, 
#                     name='{}'.format(0))

# df = df.append(new_row, ignore_index= False)

    

# PATH = "Forging_Data/Fermionic_Hamiltonian_Forging_lr_{}_complex_{}_n_qubits_{}_trial_{}.pkl".format(lr, complex_true, n_qubits, trial)
# df.to_pickle(PATH)

