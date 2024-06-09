import numpy as np
import copy

def find_Mch_seq(K, X_ijk, C_ij, t):
    Mch_seq = [[] for _ in range(K)]
    for k in range(K):
        valid_indices  = np.argwhere((C_ij > t) & (X_ijk[:, :, k] == 1))
        C_values       = C_ij[valid_indices[:, 0], valid_indices[:, 1]]
        Mch_seq[k]     = valid_indices[np.argsort(C_values)].tolist()

    return Mch_seq


def evaluate_LocalCost(d_j, C_ij, JSet):
    C_j = np.max(C_ij, axis = 0)
    T_j = np.maximum(C_j[JSet] - d_j[JSet], 0)
    cost = np.sum(T_j)
    return cost, C_j


def RightShift (Mch, id_ope_onMch, time_available, dummy_Job_seq, dummy_Mch_seq, dummy_Xijk, dummy_Sij, dummy_Cij, p_ijk, n_j):
    Oij = dummy_Mch_seq[Mch][id_ope_onMch]
    i   = Oij[0]
    j   = Oij[1]

    if time_available <= dummy_Sij[i][j]:
        return dummy_Xijk, dummy_Sij, dummy_Cij
    else:
        # Define new start time and end time of operation Oij
        if time_available > dummy_Sij[i][j]:
            dummy_Sij[i][j]   = copy.deepcopy(time_available)
            dummy_Cij[i][j]   = time_available + p_ijk[i][j][Mch]
        next_time_available   = copy.deepcopy(dummy_Cij[i][j])

        # Consider successing operation on the same Machine
        if id_ope_onMch < len(dummy_Mch_seq[Mch]) -1:
            RightShift(Mch, id_ope_onMch+1, next_time_available, dummy_Job_seq, dummy_Mch_seq, dummy_Xijk, dummy_Sij, dummy_Cij, p_ijk, n_j)
        
        # Consider successing operation on the same Job
        if i < int(n_j[j]) - 1:
            next_Oij          = [i+1, j]
            that_Mch          = np.argwhere(dummy_Xijk[i+1, j, :] == 1)[0][0]
            that_id_ope_onMch = next((i for i, sublist in enumerate(dummy_Mch_seq[that_Mch]) if sublist == next_Oij), -1)
            RightShift(that_Mch, that_id_ope_onMch, next_time_available, dummy_Job_seq, dummy_Mch_seq, dummy_Xijk, dummy_Sij, dummy_Cij, p_ijk, n_j)
        return dummy_Xijk, dummy_Sij, dummy_Cij