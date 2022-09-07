#!/usr/bin/env python3
'''
load spinfree RDM and CASPT2 intermediate tensors and demonstrate consistency with respect to:
    - hermiticity
    - partial traces
    - Fock contraction (on-the-fly contracted 4F should be equivalent to explicit contraction)

all expectation value arrays (RDMs and PT2 intermediates) are normal-ordered but indexed 
in creation-annihilation pairs i.e.
rdm1[a, i] = sum_s <a_s* i_s>
rdm2[a, i, b, j] = sum_s,t <a_s* b_t* j_t i_s>
rdm3[a, i, b, j, c, k] = sum_s,t,u <a_s* b_t* c_u* k_u j_t i_s>
rdm4f[a, i, b, j, c, k] = sum_s,t,u,v sum_d,l F_d,l <a_s* b_t* c_u* d_v* l_v k_u j_t i_s>
where the sums run over spin indices (alpha, beta), a_s signifies the annihilation of an electron in the s spin function of orbital a
and a_s* signifies the corresponing creation operator
'''

import numpy as np

# ascertain number of orbitals and electrons from the 1RDM
rdm1 = np.load('rdm1.npy')
norb = rdm1.shape[0]
nelec = np.einsum('aa->', rdm1)

# check hermiticity
assert np.allclose(rdm1, rdm1.transpose(1, 0))

def one_trace_fac(rank):
    return nelec-rank+1

def trace_fac(rank_senior, rank_junior):
    out = 1
    for rank in range(rank_junior+1, rank_senior+1): out*=one_trace_fac(rank)
    return out

# load 2RDM
rdm2 = np.load('rdm2.npy')
# check hermiticity
assert np.allclose(rdm2, rdm2.transpose(1, 0, 3, 2))
# check partial traces
assert np.allclose(np.einsum('ppqq->', rdm2), trace_fac(2, 0))
assert np.allclose(np.einsum('aipp->ai', rdm2), trace_fac(2, 1)*rdm1)
assert np.allclose(np.einsum('ppai->ai', rdm2), trace_fac(2, 1)*rdm1)

# load 3RDM
rdm3 = np.load('rdm3.npy')
# check hermiticity
assert np.allclose(rdm3, rdm3.transpose(1, 0, 3, 2, 5, 4))
# check partial traces
assert np.allclose(np.einsum('ppqqrr->', rdm3), trace_fac(3, 0))
assert np.allclose(np.einsum('aippqq->ai', rdm3), trace_fac(3, 1)*rdm1)
assert np.allclose(np.einsum('aibjpp->aibj', rdm3), trace_fac(3, 2)*rdm2)

# load 4RDM
rdm4 = np.load('rdm4.npy')
# check hermiticity
assert np.allclose(rdm4, rdm4.transpose(1, 0, 3, 2, 5, 4, 7, 6))
# check partial traces
assert np.allclose(np.einsum('ppqqrrss->', rdm4), trace_fac(4, 0))
assert np.allclose(np.einsum('aippqqrr->ai', rdm4), trace_fac(4, 1)*rdm1)
assert np.allclose(np.einsum('aibjppqq->aibj', rdm4), trace_fac(4, 2)*rdm2)
assert np.allclose(np.einsum('aibjckpp->aibjck', rdm4), trace_fac(4, 3)*rdm3)

# load on-the-fly contracted Fock-4RDM
rdm4f = np.load('rdm4f.npy')
# check hermiticity
assert np.allclose(rdm4f, rdm4f.transpose(1, 0, 3, 2, 5, 4))

# load generalized Fock matrix
fock = np.load('fock.npy')

# check contractions
assert np.allclose(rdm4f, np.einsum('aibjckdl,ai->bjckdl', rdm4, fock))
assert np.allclose(rdm4f, np.einsum('aibjckdl,bj->aickdl', rdm4, fock))
assert np.allclose(rdm4f, np.einsum('aibjckdl,ck->aibjdl', rdm4, fock))
assert np.allclose(rdm4f, np.einsum('aibjckdl,dl->aibjck', rdm4, fock))
