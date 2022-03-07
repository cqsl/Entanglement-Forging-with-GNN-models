#!/usr/bin/env python
# coding: utf-8
import netket as nk
from netket.operator.spin import sigmax,sigmaz
from netket import jax as nkjax
# import jax
import jax.numpy as jnp
# from jax import random
# from tqdm import tqdm
import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from pennylane.operation import AnyWires, Operation
from pennylane.wires import Wires

# from TFIM import *


def brick_wall_entangling(params):
    layers, qubits, _ = params.shape
    for i in range(qubits):
        qml.Hadamard(wires = i)
    for i in range(layers):
        for j in range(qubits):
            qml.Rot(*params[i][j], wires = j)
        for j in range(int(qubits/2)):
            if i%2 == 0:
                qml.CNOT(wires = [2*j, 2*j+1])
            if i%2 == 1:
                qml.CNOT(wires = [2*j + 1, (2*j+2)%qubits])
                

def classical_correlators(OP_1, OP_2, idx1, idx2, n_qubits, psi):
    if idx1 > idx2:
        raise Exception('idx1 must be smaller than idx 2')
        
    if OP_1.shape[0] > 2:
        raise Exception('OP dimesnion > 2x2 not implemented')
        
    if OP_2.shape[0] > 2:
        raise Exception('OP dimesnion > 2x2 not implemented')
        
    if idx2 > n_qubits:
        raise Exception('idx2 must be smaller equal than n_qubits')
    one = np.eye(2)
    op = [one]*(min(idx1, n_qubits-2)) + [OP_1] + [one]*(idx2- idx1-1) + [OP_2] + [one]*(n_qubits - idx2 -1)
    M = 1
    for O in op:
        M = np.kron(M, O)
    return np.inner(psi, np.matmul(M,psi))
        
    
def calculate_all_correlators_classical(n_qubits, psi, 
                                        OP_1 = qml.PauliZ(0).matrix, OP_2 = qml.PauliZ(0).matrix):
    """
    n_qubits: are here number of qubits of composite system
    """
    correlation_dict = {}
    for idx1 in range(n_qubits):
        for idx2 in range(n_qubits):
            if idx1 < idx2:
                ZZ = classical_correlators(OP_1, OP_2, idx1, idx2, n_qubits, psi)
                correlation_dict["{}{}".format(idx1, idx2)] = ZZ
    return correlation_dict  



def get_all_operators_random_old(grid_size = [2,4], seed = 42, random=True):
    if random:
        np.random.seed(seed)
        
    n_qubits = np.prod(grid_size)
    N = n_qubits//2

    if random:
        h_subsys = np.random.rand(N) # N local fields
    else:
        h_subsys = np.ones(N)

    full_graph = nk.graph.Grid(grid_size, pbc = [False, True])
    hi_full = nk.hilbert.Spin(s=1 / 2, N=full_graph.n_nodes)

    full_nodes = full_graph.nodes()
    full_edges = full_graph.edges()


    N_sub = grid_size[1]//2
    sub_graph = nk.graph.Grid([grid_size[0], N_sub], pbc = [False, False])
    hi = nk.hilbert.Spin(s=1 / 2, N=sub_graph.n_nodes)

    full_grid = np.array(list(full_nodes)).reshape(grid_size)
#     N = grid_size[1]//2

    sub_grid = full_grid[: , 0:N_sub]
    sub_grid_B = full_grid[: , N_sub:]

    overlap_edges = []
    sys_A = []
    sys_B = []
    for e in full_edges:
        if (np.isin(sub_grid, e[0])*1.0).sum() == 1. and (np.isin(sub_grid, e[1])*1.0).sum() ==1.:
            sys_A.append(e)
        elif (np.isin(sub_grid, e[0])*1.0).sum() == 0 and (np.isin(sub_grid, e[1])*1.0).sum() == 0:
            sys_B.append(e)
        else:
            overlap_edges.append(e)

    all_edges = full_graph.edges()
    for e in overlap_edges:
        all_edges.remove(e)

    grid = np.array(list(full_graph.nodes())).reshape(grid_size) #original grid numbering   
    new_grid = grid.copy()
    new_grid[: , 0:N_sub] = np.arange(N).reshape(sub_grid.shape)
    new_grid[:, N_sub:2*N_sub] = np.arange(N).reshape(sub_grid.shape)   + n_qubits//2  # new numbering with 2 subsy

    overlap_edges_new = []
    for e in overlap_edges:
        e1 = new_grid[np.where(grid == e[0])]
        e2 = new_grid[np.where(grid == e[1])]
        overlap_edges_new.append([e1.item(), e2.item()])

#     sub_graph_nodes = N_sub*grid_size[0]

    if random:
        J_subsys = np.random.rand(len(all_edges)) # There are N-1 couplings each subsystem and 1 to connect the two subsys
        J_overlap = np.random.rand(grid_size[0]) # Overlaps in same raw must be equal
    else:
        J_subsys = np.ones(len(all_edges)) # There are N-1 couplings each subsystem and 1 to connect the two subsys
        J_overlap = np.ones(grid_size[0]) # Overlaps in same raw must be equal

    H_A = [h_subsys[i]*sigmax(hi,i) for i in sub_graph.nodes()]
    H_A += [J_subsys[k]*sigmaz(hi,i)*sigmaz(hi,j) for k,(i,j) in enumerate(sub_graph.edges())]

    H_full = [h_subsys[k]*sigmax(hi_full,i) for k, i in enumerate(sub_grid.flatten())]
    H_full += [h_subsys[k]*sigmax(hi_full,i) for k,i in enumerate(sub_grid_B.flatten())]
    H_full += [J_subsys[i]*sigmaz(hi_full,sys_A[i][0])*sigmaz(hi_full,sys_A[i][1]) for i in range(len(sys_A))]
    H_full += [J_subsys[i]*sigmaz(hi_full,sys_B[i][0])*sigmaz(hi_full,sys_B[i][1]) for i in range(len(sys_B))]
    H_full += [J_overlap[k//2]*sigmaz(hi_full,i)*sigmaz(hi_full,j) for k, (i,j) in enumerate(overlap_edges)]

    Obs_AB = [[qml.PauliZ(e[0]%N)@qml.PauliZ(e[1]%N), qml.Identity(0)] for e in overlap_edges_new]

    return Obs_AB, H_A, H_full, J_overlap, J_subsys, h_subsys



def get_all_operators_random(grid_size = [2,4], seed = 42, random=True):
    if random:
        np.random.seed(seed)
        
    n_qubits = np.prod(grid_size)
    N = n_qubits//2

    if random:
        h_subsys = np.random.rand(N) # N local fields
    else:
        h_subsys = np.ones(N)

    full_graph = nk.graph.Grid(grid_size, pbc = [False, True])
    hi_full = nk.hilbert.Spin(s=1 / 2, N=full_graph.n_nodes)

    full_nodes = full_graph.nodes()
    full_edges = full_graph.edges()


    N_sub = grid_size[1]//2
    sub_graph = nk.graph.Grid([grid_size[0], N_sub], pbc = [False, False])
    hi = nk.hilbert.Spin(s=1 / 2, N=sub_graph.n_nodes)

    full_grid = np.array(list(full_nodes)).reshape(grid_size)
#     N = grid_size[1]//2

    sub_grid = full_grid[: , 0:N_sub]
    sub_grid_B = full_grid[: , N_sub:]

    overlap_edges = []
    sys_A = []
    sys_B = []
    for e in full_edges:
        if (np.isin(sub_grid, e[0])*1.0).sum() == 1. and (np.isin(sub_grid, e[1])*1.0).sum() ==1.:
            sys_A.append(e)
        elif (np.isin(sub_grid, e[0])*1.0).sum() == 0 and (np.isin(sub_grid, e[1])*1.0).sum() == 0:
            sys_B.append(e)
        else:
            overlap_edges.append(e)

    all_edges = full_graph.edges()
    for e in overlap_edges:
        all_edges.remove(e)

    grid = np.array(list(full_graph.nodes())).reshape(grid_size) #original grid numbering   
    new_grid = grid.copy()
    new_grid[: , 0:N_sub] = np.arange(N).reshape(sub_grid.shape)
    new_grid[:, N_sub:2*N_sub] = np.arange(N).reshape(sub_grid.shape)   + n_qubits//2  # new numbering with 2 subsy

    overlap_edges_new = []
    for e in overlap_edges:
        e1 = new_grid[np.where(grid == e[0])]
        e2 = new_grid[np.where(grid == e[1])]
        overlap_edges_new.append([e1.item(), e2.item()])

#     sub_graph_nodes = N_sub*grid_size[0]

    if random:
        J_subsys = np.random.rand(len(all_edges)) # There are N-1 couplings each subsystem and 1 to connect the two subsys
        J_overlap = np.random.rand(grid_size[0]) # Overlaps in same raw must be equal
    else:
        J_subsys = np.ones(len(all_edges)) # There are N-1 couplings each subsystem and 1 to connect the two subsys
        J_overlap = np.ones(grid_size[0]) # Overlaps in same raw must be equal

    H_A = [h_subsys[i]*sigmax(hi,i) for i in sub_graph.nodes()]
    H_A += [J_subsys[k]*sigmaz(hi,i)*sigmaz(hi,j) for k,(i,j) in enumerate(sub_graph.edges())]

    H_full = [h_subsys[k]*sigmax(hi_full,i) for k, i in enumerate(sub_grid.flatten())]
    H_full += [h_subsys[k]*sigmax(hi_full,i) for k,i in enumerate(sub_grid_B.flatten())]
    H_full += [J_subsys[i]*sigmaz(hi_full,sys_A[i][0])*sigmaz(hi_full,sys_A[i][1]) for i in range(len(sys_A))]
    H_full += [J_subsys[i]*sigmaz(hi_full,sys_B[i][0])*sigmaz(hi_full,sys_B[i][1]) for i in range(len(sys_B))]
    H_full += [J_overlap[k//2]*sigmaz(hi_full,i)*sigmaz(hi_full,j) for k, (i,j) in enumerate(overlap_edges)]

    OP = []
    for j, e in enumerate(overlap_edges_new):
        for i in range(N):
            if i ==0:
                if e[0]%N != i:
                    OP_A = qml.Identity(i)
                else:
                    OP_A = qml.PauliZ(e[0]%N)
                if e[1]%N != i:
                    OP_B = qml.Identity(i)
                else:
                    OP_B = qml.PauliZ(e[1]%N)
            else:
                if e[0]%N != i:
                    OP_A = OP_A@qml.Identity(i)
                else:
                    OP_A = OP_A@qml.PauliZ(e[0]%N)
                if e[1]%N != i:
                    OP_B = OP_B@qml.Identity(i)
                else:
                    OP_B = OP_B@qml.PauliZ(e[1]%N)
        OP.append([J_overlap[j//2], OP_A, OP_B])
    

    return OP, H_A, H_full, J_overlap, J_subsys, h_subsys


class QubitUnitary(Operation):
    num_wires = AnyWires
    num_params = 1
    grad_method = None

    def __init__(self, *params, wires, do_queue=True):
        wires = Wires(wires)

        super().__init__(*params, wires=wires, do_queue=do_queue)

    @classmethod
    def _matrix(cls, *params):
        return params[0]


    def adjoint(self):
        return QubitUnitary(qml.math.T(qml.math.conj(self.matrix)), wires=self.wires)


    def label(self, decimals=None, base_label=None):
        return super().label(decimals=decimals, base_label=base_label or "U")


def apply_unitary(M, wires):
    QubitUnitary(M, wires=wires)