import pandas as pd
import numpy as np
import copy
import random
from util.util_action import find_Mch_seq

def find_indices(X, k):
    indices = np.argwhere(X[:, :, k] == 1)
    return [(idx[0], idx[1]) for idx in indices]

def find_avgload(J, K, h_ijk, p_ijk, n_j):
    avgload_k   = np.zeros((K)) # average load
    denominator = np.zeros((J))
    part        = np.zeros((J))

    for k in range(K):
        for j in range(J):
            denominator [j]   = sum(h_ijk[i][j][k] for i in range(int(n_j[j])))
            if denominator[j] == 0:
                part[j] = 0
            else:
                part[j] = sum(h_ijk[i][j][k] * p_ijk[i][j][k] for i in range(int(n_j[j])))/ sum(h_ijk[i][j][k] for i in range(int(n_j[j])))

        avgload_k [k] = sum(part[j] for j in range(J))
        
    return avgload_k


def find_P(J, K, h_ijk, p_ijk, n_j):
    P = sum(sum(sum(h_ijk[i][j][k]*p_ijk[i][j][k] for k in range(K))/sum(h_ijk[i][j][k] for k in range(K)) for i in range(int(n_j[j])))/int(n_j[j]) for j in range(J)) / J
    
    return P

def find_S_j (t, JSet, ODSet, C_ij, J):
    S_j = np.zeros(J)
    # Create a boolean mask for jobs with empty ODSet
    empty_mask = np.array([len(ODSet[j]) == 0 for j in JSet])

    # Handle the non-empty and empty cases
    non_empty_jobs = np.array(JSet)[~empty_mask]
    if non_empty_jobs.size > 0:
        last_operations = np.array([ODSet[j][-1] for j in non_empty_jobs])
        max_C_ij = np.max(C_ij[last_operations, non_empty_jobs], axis=0)
        S_j[non_empty_jobs] = np.maximum(max_C_ij, t)

    # Set S_j values for jobs with empty ODSet
    S_j[np.array(JSet)[empty_mask]] = t

    return S_j

def generate_nonzero_exponential(scale):
    value = 0
    while value == 0:
        value = round(np.random.exponential(scale=scale))
    return value


def change_dataset(p_ijk, h_ijk, X_ijk, S_ij, C_ij, MC_ji, n_MC_ji, n_j, j, i, processed, I, K, org_p_ijk, org_h_ijk):
    """p_ijk, h_ijk, X_ijk, S_ij, C_ij"""
    slice_p_ijk   = p_ijk[:, j, :].copy()
    slice_h_ijk   = h_ijk[:, j, :].copy()
    slice_X_ijk   = X_ijk[:, j, :].copy()
    slice_S_ij    = S_ij [:, j]   .copy()
    slice_C_ij    = C_ij [:, j]   .copy()

    # Check if need modify shape
   
    if n_j[j] < slice_p_ijk.shape[0]:
        # Move rows down
        dummy_p = slice_p_ijk.copy()
        dummy_h = slice_h_ijk.copy()
        dummy_X = slice_X_ijk.copy()
        dummy_S = slice_S_ij .copy()
        dummy_C = slice_C_ij .copy()

        for operation in range(i, int(n_j[j])):
            dummy_p[operation + 1] = copy.deepcopy(slice_p_ijk[operation])
            dummy_h[operation + 1] = copy.deepcopy(slice_h_ijk[operation])
            dummy_X[operation + 1] = copy.deepcopy(slice_X_ijk[operation])
            dummy_S[operation + 1] = copy.deepcopy(slice_S_ij [operation])
            dummy_C[operation + 1] = copy.deepcopy(slice_C_ij [operation])

        slice_p_ijk = copy.deepcopy(dummy_p)
        slice_h_ijk = copy.deepcopy(dummy_h)
        slice_X_ijk = copy.deepcopy(dummy_X)  
        slice_S_ij  = copy.deepcopy(dummy_S)
        slice_C_ij  = copy.deepcopy(dummy_C)

        # Update row i+1 with max(0, row index i - processed)
        slice_p_ijk[i+1] = np.maximum(0, slice_p_ijk[i] - processed)
        slice_S_ij [i+1] = slice_S_ij[i] + processed
        slice_C_ij [i]   = slice_S_ij[i] + processed

        p_ijk[:, j, :] = slice_p_ijk.copy()
        h_ijk[:, j, :] = slice_h_ijk.copy()
        X_ijk[:, j, :] = slice_X_ijk.copy()
        S_ij [:, j]    = slice_S_ij .copy()
        C_ij [:, j]    = slice_C_ij .copy()
    else:
        # Insert a new row after row index i
        new_row_p      = np.maximum(0, slice_p_ijk[i] - processed)
        new_row_h      = slice_h_ijk  [i].copy()
        new_row_X      = slice_X_ijk  [i].copy()
        new_ele_S      = slice_S_ij   [i] + processed
        cur_ele_C      = slice_S_ij   [i] + processed

        slice_p_ijk    = np.insert(slice_p_ijk,   i+1, new_row_p, axis=0)
        slice_h_ijk    = np.insert(slice_h_ijk,   i+1, new_row_h, axis=0)
        slice_X_ijk    = np.insert(slice_X_ijk,   i+1, new_row_X, axis=0)
        slice_S_ij     = np.insert(slice_S_ij,    i+1, new_ele_S, axis=0)
        slice_C_ij     = np.insert(slice_C_ij,    i,   cur_ele_C, axis=0)

        p_ijk          = np.pad(p_ijk,     ((0, 1), (0, 0), (0, 0)), mode='constant', constant_values=0)
        h_ijk          = np.pad(h_ijk,     ((0, 1), (0, 0), (0, 0)), mode='constant', constant_values=0)
        X_ijk          = np.pad(X_ijk,     ((0, 1), (0, 0), (0, 0)), mode='constant', constant_values=0)
        S_ij           = np.pad(S_ij,      ((0, 1), (0, 0)),         mode='constant', constant_values=0)
        C_ij           = np.pad(C_ij,      ((0, 1), (0, 0)),         mode='constant', constant_values=0)

        p_ijk[:, j, :] = slice_p_ijk  .copy()
        h_ijk[:, j, :] = slice_h_ijk  .copy()
        X_ijk[:, j, :] = slice_X_ijk  .copy()
        S_ij [:, j]    = slice_S_ij   .copy()
        C_ij [:, j]    = slice_C_ij   .copy()

        org_p_ijk      = np.pad(org_p_ijk, ((0, 1), (0, 0), (0, 0)), mode='constant', constant_values=0)
        org_h_ijk      = np.pad(org_h_ijk, ((0, 1), (0, 0), (0, 0)), mode='constant', constant_values=0)

        I += 1

    for k in range(K):
        p_ijk[i,j,k] = processed if p_ijk[i,j,k] != 999 else 999

    """MC_ji and n_MC_ji"""
    sublist_to_copy1   = copy.deepcopy(MC_ji[j][i])
    sublist_to_copy2   = copy.deepcopy(n_MC_ji[j][i])
    MC_ji  [j].insert(i + 1, sublist_to_copy1)
    n_MC_ji[j].insert(i + 1, sublist_to_copy2)

    """n_j"""
    n_j[j] += 1
    
    return p_ijk, h_ijk, X_ijk, S_ij, C_ij, MC_ji, n_MC_ji, n_j, I, org_p_ijk, org_h_ijk


def random_events(t, K, X_ijk, S_ij, C_ij, JA_event, MB_event, events, maxtimeacceptnewjob, J, defectProb):
    
    # """Machine breakdown"""
    # next_MB_time   = float('inf')
    # Mch_seq        = find_Mch_seq(K, X_ijk, C_ij, t)
    # valid_sublists = [(index, sublist) for index, sublist in enumerate(Mch_seq) if sublist]
    # if len(valid_sublists) >1:
    #     # Randomly select a sublist from valid_sublists
    #     index, machine_BD = random.choice(valid_sublists)
    #     # Extract BD_ranges directly using list comprehension
    #     BD_ranges         = [range_ for i, j in machine_BD for range_ in [(S_ij[i][j], C_ij[i][j])] if range_]
    #     # Randomly choose a BD_range from BD_ranges
    #     chosen_BD_range   = random.choice(BD_ranges)
    #     # Generate a random integer within the chosen range [start_BD, end_BD]
    #     next_MB_time      = random.randint(*chosen_BD_range)

    #     if next_MB_time not in events:
    #         events[next_MB_time] = []
    #     new_event = ("MB", index)
    #     events[next_MB_time].append(new_event)

    # """Job arrival"""
    # if t < maxtimeacceptnewjob:
    #     C_j           = np.max(C_ij, axis=0)
    #     after_mask    = C_j > t
    #     filtered_job  = C_j[after_mask]
    #     prob          = np.random.random(len(filtered_job))

    #     defected_mask = prob < defectProb
    #     defected_job  = filtered_job[defected_mask]
 
    #     JA_list       = []
    #     if len(defected_job) > 0:
    #         indices_after_mask = np.where(after_mask)[0]
    #         defected_jobs      = indices_after_mask[defected_mask]
    #         C_defected_jobs    = C_j[defected_jobs]
    #         next_JA_time       = np.min(C_defected_jobs)
    #         if next_JA_time <= next_MB_time:
    #             soonest_mask   = C_defected_jobs == next_JA_time
    #             JA_list        = defected_jobs[soonest_mask].tolist()           
        
    #     if JA_list:
    #         if next_JA_time not in events:
    #             events[next_JA_time] = []
    #         formatted_elements = [("JA", element) for element in JA_list]
    #         events[next_MB_time].extend(formatted_elements)

    jobresemble = random.choice(list(range(J)))
    events = {20: [("MB", 1)], 
              50: [("MB", 3)], 
              75: [("MB", 5)], 
              90: [("JA", jobresemble)], 
              110: [("MB", 4)]}
    filtered_dict       = {key: value for key, value in events.items() if key > t}
    if filtered_dict:
        t               = min(filtered_dict)
        triggered_event = filtered_dict[t].copy()
    else: 
        t               = np.max(C_ij)
        triggered_event = None
       

    return JA_event, MB_event, events, t, triggered_event

    
def snapshot(t, triggered_event, MC_ji, n_MC_ji,                 \
             d_j, n_j, p_ijk, h_ijk, J, I, K, X_ijk, S_ij, C_ij, \
             OperationPool, MB_info, MB_record, S_k,             \
             org_J, org_p_ijk, org_h_ijk, org_n_j,               \
             org_MC_ji, org_n_MC_ji, C_j                         ):
    
    # Set ----------------------------------------------------------------
    ## Job and operation still need to be scheduled
    LS_j  = np.max(S_ij,axis=0)
    JSet  = np.where(LS_j >= t)[0].tolist()
    OJSet = [[] for _ in range(J)]
    for j in JSet:
        OJSet[j] = np.where(S_ij[:int(n_j[j]), j] >= t)[0].tolist()

    ## Job and operation has been run
    ES_j  = np.min(S_ij, axis=0)
    DSet  = np.where(ES_j < t)[0].tolist()
    ODSet = [[] for _ in range(J)]
    for j in DSet:
        ODSet[j] = np.where(S_ij[:int(n_j[j]), j] < t)[0].tolist()

    #------------------------------------------------------------------------
    JA                 = []
    MB                 = []
    Oij_on_machine     = [[] for _ in range(K)]
    affected_Oij       = {}
    re                 = np.zeros((K))          # repair time

    if triggered_event is not None:
        for type, detailed in triggered_event:
            if type == "MB": MB.append(detailed)    # if Machine break down
            else:            JA.append(detailed)    # if Job arrival
                
        # Operation on machine ------------------------------------------------
        for k in range(K):
            i, j = np.where((X_ijk[:, :, k] == 1) & (S_ij < t) & (C_ij > t))
            if len(i) != 0 and len(j) != 0:
                operation = (i,j)
                Oij_on_machine[k].append(operation)

        # Check if MB -----------------------------------------------------------
        if MB:
            X_mask =  X_ijk.astype(bool)
            for k in MB:
                re[k] = random.randint(5, 60)
                """Add in MB_record"""
                if k not in MB_record:
                    MB_record[k] = []
                record = (t, t+re[k])
                MB_record[k].append(record)
                """Find affected operation"""
                # Use the boolean mask to find the indices where overlap occurs
                overlap_mask = np.logical_or(
                    np.logical_and(S_ij >= t        , C_ij <= t + re[k]),       # in
                    np.logical_and(S_ij <= t        , C_ij >  t        ),       # left
                    np.logical_and(S_ij <  t + re[k], C_ij >= t + re[k]))       # right
                indices = np.argwhere(X_mask[:, :, k] & overlap_mask[:, :])
                # Append the indices to the affected_Oij dictionary
                if len(indices) > 0:
                    affected_Oij[k] = indices.tolist()

        # Soonest start time ------------------------------------------------------
        id       = np.zeros((K)) # boolean, =1 if machine idle at time t, 0 otherwise (busy processing or repair)
        bu       = np.zeros((K)) # boolean, =1 if machine busy processing, = 0 if machine is repaired
        av       = np.zeros((K)) # remaining processing time 

        for k in range(K):
            if k not in MB:
                operation = Oij_on_machine[k]
                if len(operation) != 0:
                    i = copy.deepcopy(operation[0][0][0])
                    j = copy.deepcopy(operation[0][1][0])
                    av[k] = C_ij[i][j] - t
                    bu[k] = 1
                else:   
                    id[k] = 1
            else:
                operation = Oij_on_machine[k]
                if len(operation) != 0:
                    i = copy.deepcopy(operation[0][0][0])
                    j = copy.deepcopy(operation[0][1][0])
                    processed  = t - S_ij[i][j]
                
                    """Break the operation into  2 segments"""
                    p_ijk, h_ijk, X_ijk, S_ij, C_ij, MC_ji, n_MC_ji, n_j, I, org_p_ijk, org_h_ijk = change_dataset(p_ijk, h_ijk, X_ijk, S_ij, C_ij, MC_ji, n_MC_ji, n_j, j, i, processed, I, K, org_p_ijk, org_h_ijk)
                    OJSet[j].insert(0, i)                     # need to schedule the remaining of affected operation
                    OJSet[j]   = [ops+1 for ops in OJSet[j]]  # update OJSet to match with parameters stored in np array
                    ODSet[j].append(i)
                    if j not in JSet:
                        JSet.append(j)
                    if j not in DSet:
                        DSet.append(j)

        S_k = np.maximum(t + (1-id)*(bu*av + (1-bu)*re), S_k)
        
        # Check if JA -----------------------------------------------------------
        if JA: 
            for jobresemble in JA: 
                """Adjust the dataset"""              
                J += 1
                # num operation of new job
                n_newjob                   = copy.deepcopy(org_n_j[jobresemble])
                n_j                        = np.append(n_j, n_newjob)
                # processing time
                p_newjob                   = copy.deepcopy(org_p_ijk[:, jobresemble, :])
                p_newjob_reshape           = copy.deepcopy(p_newjob[:, np.newaxis, :])
                p_ijk                      = np.concatenate((p_ijk, p_newjob_reshape), axis= 1)
                # deadline
                d_newjob                   = t + np.round(np.sum(np.mean(p_newjob, axis=1))*np.random.uniform(1, 5))
                d_j                        = np.append(d_j, d_newjob)
                # capable machine            
                h_newjob                   = copy.deepcopy(org_h_ijk[:, jobresemble, :])
                h_newjob_reshape           = copy.deepcopy(h_newjob[:, np.newaxis, :])
                h_ijk                      = np.concatenate((h_ijk, h_newjob_reshape), axis= 1)

                MC_newjob                  = copy.deepcopy(org_MC_ji[jobresemble])
                n_MC_newjob                = copy.deepcopy(org_n_MC_ji[jobresemble])
                MC_ji  .append(MC_newjob)
                n_MC_ji.append(n_MC_newjob)

                # Adjust the org
                org_n_j                    = np.append(org_n_j, n_newjob)
                org_p_ijk                  = np.concatenate((org_p_ijk, p_newjob_reshape), axis= 1)
                org_h_ijk                  = np.concatenate((org_h_ijk, h_newjob_reshape), axis= 1)
                org_MC_ji.append(MC_newjob)
                org_n_MC_ji.append(n_MC_newjob)

                new_row_values                        = [J-1, n_j[J-1]]
                OperationPool.loc[len(OperationPool)] = new_row_values

                JSet.append(J-1)
                OJSet.append(list(range(int(n_newjob))))
                ODSet.append([])

                # X, S, C
                X_ijk          = np.pad(X_ijk,     ((0, 0), (0, 1), (0, 0)), mode='constant', constant_values=0)
                S_ij           = np.pad(S_ij,      ((0, 0), (0, 1)),         mode='constant', constant_values=0)
                C_ij           = np.pad(C_ij,      ((0, 0), (0, 1)),         mode='constant', constant_values=0)
                C_j            = np.pad(C_j,       ((0, 1)),                 mode='constant', constant_values=0)
                """"Check urgency"""
                p_excluded     = p_newjob[1:int(n_newjob -1)]
                LC_firstope    = d_newjob - np.sum(np.mean(p_excluded, axis=1))

                Acceptable     = [k for k in MC_newjob[0] if S_k[k] + p_newjob[0][k] <= LC_firstope]
                if len(Acceptable) == 0:
                    # Find LC of operation currently on machine k
                    LC_operation_on_k = np.full(K, np.inf)
                    for k in MC_newjob[0]:
                        operation = copy.deepcopy(Oij_on_machine[k])

                        if len(operation) != 0:
                            i = copy.deepcopy(operation[0][0][0])
                            j = copy.deepcopy(operation[0][1][0])

                            # Calculate LC of the operation
                            mean_p = np.mean(p_ijk[:, j, :], axis=1)
                            LC_operation_on_k[k] = d_j[j] - np.sum(mean_p[i+1 : int(n_j[j]-1)])

                    # Consider to remove current operation on machine
                    Considered_list = [k for k in MC_newjob[0] if LC_firstope < LC_operation_on_k[k] and LC_operation_on_k[k] != np.inf]
                    if len(Considered_list) != 0:
                        sums        = S_k[Considered_list] + p_newjob[0, Considered_list]
                        k_min_index = np.argmin(sums)
                        k           = copy.deepcopy(Considered_list[k_min_index])
                        if Oij_on_machine[k]:
                            operation  = copy.deepcopy(Oij_on_machine[k]) ###################
                            i, j       = copy.deepcopy(operation)
                            processed  = t - S_ij[i][j]

                            """Break the operation into  2 segments"""
                            p_ijk, h_ijk, X_ijk, S_ij, C_ij, MC_ji, n_MC_ji, n_j, I, org_p_ijk, org_h_ijk = change_dataset(p_ijk, h_ijk, X_ijk, S_ij, C_ij, MC_ji, n_MC_ji, n_j, j, i, processed, I, K, org_p_ijk, org_h_ijk)
                            OJSet[j]   = [ops+1 for ops in OJSet[j]]  # update OJSet to match with parameters stored in np array
                            OJSet[j].insert(0, i)                     # need to schedule the remaining of affected operation
                            ODSet[j].append(i)
                            if j not in JSet:
                                JSet.append(j)
                            if j not in DSet:
                                DSet.append(j)

        S_j = find_S_j (t, JSet, ODSet, C_ij, J) 
    
    else:
        S_j = np.zeros((J))
    # Number of operation of job j that have been run ---------------------------------
    n_ops_left_j = np.array([len(sublist) for sublist in OJSet])
    OperationPool['Num operation left'] = n_ops_left_j.astype(int)

    # Observation --------------------------------------------------------------------    
    # Update remaining repair time
    
    if MB_info.shape != (0,) and MB_info.shape!= (3,):
        MB_info[1]         = t - MB_info[2]
        notDoneRepair_mask = MB_info[1, :] > 0
        MB_info            = MB_info[:, notDoneRepair_mask]
    else:
        if MB_info.shape == (3,):
            MB_info[1] = t - MB_info[2]
            MB_info = np.zeros((0)) if MB_info[1] <= 0 else MB_info

    # Add new breakdown machine
    if MB:
        k = copy.deepcopy(MB[0])
        new_column = np.array([k, re[k], S_k[k]])
        MB_info = new_column if MB_info.shape == (0,) and MB_info.shape != (3,) else np.column_stack((MB_info, new_column))
 
        for k in MB[1:]:
            new_column = np.array([k, re[k], S_k[k]])
            MB_info = np.column_stack((MB_info, new_column))
    remaining_re = np.mean(MB_info[1]) if MB_info.shape != (0,) else 0

    percent_JA = len(JA)/org_J 
    if MB_info.shape == (0,) :
        percent_MB = 0 
    elif MB_info.shape == (3,):
        percent_MB = 1/K
    else:
        percent_MB = MB_info.shape[1]/K

    JA = 1 if JA else 0
    MB = 1 if MB else 0

    #------------------------------------------------------
    # Find completion time of last operation assigned to machine k at rescheduling point t
    CT_k = np.zeros(K)
    for k in range(K):  
        indices = np.argwhere((X_ijk[:, :, k] == 1) & (S_ij < t))  
        if len(indices) > 0:
            max_index = np.argmax(S_ij[indices[:, 0], indices[:, 1]])
            i, j      = indices[max_index]
            CT_k[k]   = C_ij[i, j] 

    # The average completion time of the last operation already run on each machine at current decision point t
    T_cur = np.mean(S_k)
    
    # Tard_job
    Tard_job = [j for j in JSet if d_j[j] < T_cur]

    return  S_k, S_j, J, I, JSet, OJSet, DSet, ODSet, OperationPool,             \
            n_ops_left_j, MC_ji, n_MC_ji, d_j, n_j, p_ijk, h_ijk,                \
            org_p_ijk, org_h_ijk, org_n_j, org_MC_ji, org_n_MC_ji,               \
            Oij_on_machine, affected_Oij, MB_info, MB_record, X_ijk, S_ij, C_ij, \
            JA, percent_JA, MB, percent_MB, remaining_re, CT_k, T_cur, Tard_job, C_j


def store_schedule(X_ijk, S_ij, C_ij):
    X_previous = X_ijk.copy()
    S_previous = S_ij .copy()
    C_previous = C_ij .copy()
    return X_previous, S_previous, C_previous


def update_schedule(DSet, ODSet, X_ijk, S_ij, C_ij, X_previous, S_previous, C_previous):
    for j in DSet:
        for i in ODSet[j]:
            X_ijk[i][j] = copy.deepcopy(X_previous[i][j])
            S_ij [i][j] = copy.deepcopy(S_previous[i][j])
            C_ij [i][j] = copy.deepcopy(C_previous[i][j])
    return X_ijk, S_ij, C_ij