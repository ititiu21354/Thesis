import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy  as np
import copy
import random
from time                   import time
from pulp                   import *
from env_action.metaheu     import GeneticAlgorithm
from util.util_action       import find_Mch_seq, evaluate_LocalCost, RightShift
from scipy.optimize         import linear_sum_assignment


def action_space(J, I, K, p_ijk, h_ijk, d_j, n_j, 
                 MC_ji, n_MC_ji, n_ops_left_j, OperationPool, S_k, S_j, 
                 JSet, OJSet, Oij_on_machine, affected_Oij, 
                 t, X_ijk, S_ij, C_ij, C_j, CT_k, T_cur, Tard_job, 
                 maxtime):
    method = Method(J, I, K, p_ijk, h_ijk, d_j, n_j, 
                    MC_ji, n_MC_ji, n_ops_left_j, OperationPool, S_k, S_j, 
                    JSet, OJSet, Oij_on_machine, affected_Oij, 
                    t, X_ijk, S_ij, C_ij, C_j, CT_k, T_cur, Tard_job, 
                    maxtime)
    return [
          method.exact
        , method.GA
        , method.LFOH
        , method.LAPH
        , method.LAP_LFO
        , method.CDR1
        , method.CDR2
        # , method.CDR3
        , method.CDR4
        # , method.CDR5
        # , method.CDR6
        # , method.RouteChange_RightShift
    ]


class Method:
    def __init__(self, J, I, K, p_ijk, h_ijk, d_j, n_j, 
                 MC_ji, n_MC_ji, n_ops_left_j, OperationPool, S_k, S_j, 
                 JSet, OJSet, Oij_on_machine, affected_Oij, 
                 t, X_ijk, S_ij, C_ij, C_j, CT_k, T_cur, Tard_job,
                 maxtime):
        
        self.J     			= copy.deepcopy(J)
        self.I 	    		= copy.deepcopy(I)
        self.K  	   		= copy.deepcopy(K)
        self.p_ijk          = copy.deepcopy(p_ijk)
        self.h_ijk          = copy.deepcopy(h_ijk)
        self.d_j            = copy.deepcopy(d_j)
        self.n_j            = copy.deepcopy(n_j)
        self.MC_ji          = copy.deepcopy(MC_ji)
        self.n_MC_ji        = copy.deepcopy(n_MC_ji)
        self.n_ops_left_j   = copy.deepcopy(n_ops_left_j)
        self.OperationPool  = copy.deepcopy(OperationPool)
        self.S_k            = copy.deepcopy(S_k)
        self.S_j            = copy.deepcopy(S_j)
        self.JSet           = copy.deepcopy(JSet)
        self.OJSet          = copy.deepcopy(OJSet)
        self.Oij_on_machine = copy.deepcopy(Oij_on_machine)
        self.affected_Oij   = copy.deepcopy(affected_Oij)
        self.X_ijk          = copy.deepcopy(X_ijk)
        self.S_ij           = copy.deepcopy(S_ij)
        self.C_ij           = copy.deepcopy(C_ij)
        self.C_j            = copy.deepcopy(C_j)
        self.CT_k           = copy.deepcopy(CT_k)
        self.T_cur          = copy.deepcopy(T_cur)
        self.Tard_job       = copy.deepcopy(Tard_job)
        self.t              = copy.deepcopy(t)
        self.maxtime        = copy.deepcopy(maxtime)


    def exact(self):
        warnings.filterwarnings('ignore', 'Spaces are not permitted in the name')

        M = 10000
        # Create the LP problem
        problem = LpProblem("Flexible Job Shop Problem", LpMinimize)

        # Decision variables
        X_ijk= [[[LpVariable(f"X_{i}_{j}_{k}", cat='Binary') for k in range(self.K)] for j in range(self.J)] for i in range(self.I)]
        Y_ijab= [[[[LpVariable(f"Y_{i}_{j}_{a}_{b}", cat='Binary') for b in range(self.J)] for a in range(self.I)] for j in range(self.J)] for i in range(self.I)]

        S_ij= [[LpVariable(f"S_{i}_{j}", lowBound=0, cat='Integer') for j in range(self.J)] for i in range(self.I)]
        C_ij= [[LpVariable(f"C_{i}_{j}", lowBound=0, cat='Integer') for j in range(self.J)] for i in range(self.I)]
        C_j= [LpVariable(f"C_{j}", lowBound=0, cat='Integer') for j in range(self.J)]
        T_j= [LpVariable(f"T_{j}", lowBound=0, cat='Integer') for j in range(self.J)]

        # Constraints
        for j in self.JSet:
            for i in self.OJSet[j]:
                problem += lpSum(X_ijk[i][j][k] for k in self.MC_ji[j][i]) == 1
                problem += S_ij[i][j] >= lpSum(X_ijk[i][j][k]*self.S_k[k] for k in self.MC_ji[j][i])
                problem += S_ij[i][j] >= self.S_j[j]

                if i != int(self.n_j[j]) - 1:
                    problem += S_ij[i][j] + lpSum(X_ijk[i][j][k] * self.p_ijk[i][j][k] for k in self.MC_ji[j][i]) <= S_ij[i+1][j]

                for b in self.JSet:
                    for a in self.OJSet[b]:
                        if j < b:
                            MC_common = set(self.MC_ji[j][i]).intersection(set(self.MC_ji[b][a]))
                            for k in MC_common:
                                problem += S_ij[i][j] >= S_ij[a][b] + self.p_ijk[a][b][k] - M * (2 - X_ijk[i][j][k] - X_ijk[a][b][k] + Y_ijab[i][j][a][b])
                                problem += S_ij[a][b] >= S_ij[i][j] + self.p_ijk[i][j][k] - M * (3 - X_ijk[i][j][k] - X_ijk[a][b][k] - Y_ijab[i][j][a][b])
            
                problem += C_ij[i][j] == S_ij[i][j] + lpSum(X_ijk[i][j][k] * self.p_ijk[i][j][k] for k in self.MC_ji[j][i])
            problem += C_j[j] == C_ij[int(self.n_j[j])-1][j]
            problem += T_j[j] >= C_j[j] - self.d_j[j]

   
        problem.setObjective(lpSum([T_j[j] for j in self.JSet]))
        solver = PULP_CBC_CMD(timeLimit=self.maxtime, msg=False)
        problem.solve(solver)
        # Print the solution status
        print("Solution Status:", LpStatus[problem.status])

        if LpStatus[problem.status] == "Not Solved":
            objective_value, X_values, S_values, C_values, Cj_values = self.CDR4()
            objective_value = 10**9 # Punishment
        else:
            # Retrieve the objective value
            objective_value = value(problem.objective)

            X_values  = [[[value(X_ijk[i][j][k]) for k in range(self.K)] for j in range(self.J)] for i in range(self.I)]
            S_values  = [[value(S_ij[i][j]) for j in range(self.J)] for i in range(self.I)]
            C_values  = np.array([[value(C_ij[i][j]) for j in range(self.J)] for i in range(self.I)])

            X_values  = np.array(X_values)
            S_values  = np.array(S_values)
            C_values  = np.array(C_values)

            X_values[X_values == None] = 0.0
            S_values[S_values == None] = -999
            C_values[C_values == None] = -999

            Cj_values = np.max(C_values, axis = 0)

        return objective_value, X_values, S_values, C_values, Cj_values


    def GA(self):
        GBest, X_ijk, S_ij, C_ij, C_j = GeneticAlgorithm(self.S_k, self.S_j, self.JSet, self.OJSet, 
                                                    self.J, self.I, self.K, 
                                                    self.p_ijk, self.h_ijk, self.d_j, self.n_j, self.n_ops_left_j, 
                                                    self.MC_ji, self.n_MC_ji, self.OperationPool,
                                                    self.maxtime)
        return GBest, X_ijk, S_ij, C_ij, C_j


    def LFOH (self):
        ORSet            = []
        ORDict           = {} # key: ready time, value: [i, j]
        X_ijk            = np.zeros((self.I, self.J, self.K))
        S_ij             = np.zeros((self.I, self.J))
        C_ij             = np.zeros((self.I, self.J))
        available_time_k = self.S_k.copy()

        Reschedule_completion = False
        # Start
        for j in self.JSet:
            ready_time = self.S_j[j]
            operation  = [self.OJSet[j][0], j]
            if ready_time not in ORDict:
                ORDict[ready_time] = []
            ORDict[ready_time].append(operation)

        ready_time       = min(ORDict)
        ORSet            = ORDict[ready_time]
        sorted_operation = sorted(ORSet, key=lambda x: (self.n_MC_ji[x[1]][x[0]], self.d_j[x[1]], np.mean(self.p_ijk[x[0]][x[1]])))


        while Reschedule_completion == False:
            for i, j in sorted_operation:
                # Select machine
                if self.n_MC_ji[j][i] == 1:
                    k = self.MC_ji[j][i]
                    X_ijk[i][j][k] = 1
                else:
                    mask            = np.multiply(self.p_ijk[i][j], self.h_ijk[i][j])
                    mask[mask==0]   = 999
                    sum_values      = mask + np.maximum(ready_time, available_time_k)
                    k               = np.argmin(sum_values)
                    X_ijk[i][j][k]  = 1  
                # Calculate Start time, Completion time, and set new S_k
                S_ij[i][j]          = max(ready_time, available_time_k[k])
                C_ij[i][j]          = S_ij[i][j] + self.p_ijk[i][j][k]
                available_time_k[k] = copy.deepcopy(C_ij[i][j])
                # Adjust the set and append the dictionary (if any)
                self.OJSet[j].remove(i)
                if len(self.OJSet[j]) != 0:
                    new_ready_time = C_ij[i][j]
                    operation      = [self.OJSet[j][0], j]
                    if new_ready_time not in ORDict:
                        ORDict[new_ready_time] = []
                    ORDict[new_ready_time].append(operation)
            # Process the dictionary, check flag
            ORDict.pop(ready_time)
            if ORDict:
                ready_time       = min(ORDict)
                ORSet            = ORDict[ready_time]
                sorted_operation = sorted(ORSet, key=lambda x: (self.n_MC_ji[x[1]][x[0]], self.d_j[x[1]], np.mean(self.p_ijk[x[0]][x[1]])))
            else: Reschedule_completion = True

        # Calculate GBest
        GBest, C_j = evaluate_LocalCost(self.d_j, C_ij, self.JSet)

        return GBest, X_ijk, S_ij, C_ij, C_j


    def LAPH(self):
        ORSet            = []
        ORDict           = {} # key: ready time, value: [i, j]
        X_ijk            = np.zeros((self.I, self.J, self.K))
        S_ij             = np.zeros((self.I, self.J))
        C_ij             = np.zeros((self.I, self.J))
        dummy_JSet       = copy.deepcopy(self.JSet)
        dummy_OJSet      = copy.deepcopy(self.OJSet)

        Reschedule_completion = False
        # Start
        for j in dummy_JSet:
            ready_time = self.S_j[j]
            operation  = [dummy_OJSet[j][0], j]
            if ready_time not in ORDict:
                ORDict[ready_time] = []
            ORDict[ready_time].append(operation)

        ready_time = min(ORDict)
        ORSet      = ORDict[ready_time]

        while Reschedule_completion == False:
            # Initialize cost matrix with a high value
            num_operations = len(ORSet)
            cost_matrix    = np.full((num_operations, self.K), 999)

            # Populate the cost matrix based on constraints
            for op_idx, (i, j) in enumerate(ORSet):
                for k in range(self.K):
                    if self.h_ijk[i][j][k] == 1:
                        cost_matrix[op_idx, k] = self.S_k[k] + self.p_ijk[i][j][k]

            # Solve the assignment problem using the Hungarian algorithm
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            # Output the assignment results
            assignments = [(ORSet[row], machine) for row, machine in zip(row_ind, col_ind) if cost_matrix[row, machine] < 999]
            new_availabletime = []
            ORDict.pop(ready_time)
            for (i, j), k in assignments:
                X_ijk[i][j][k]      = 1
                S_ij[i][j]          = max(ready_time, self.S_k[k])
                C_ij[i][j]          = S_ij[i][j] + self.p_ijk[i][j][k]
                self.S_k[k]         = copy.deepcopy(C_ij[i][j])
                new_availabletime.append(self.S_k[k])
                dummy_OJSet[j].remove(i) 
                
                if len(dummy_OJSet[j]) != 0:
                    new_ready_time = copy.deepcopy(C_ij[i][j])
                    operation      = [dummy_OJSet[j][0], j]
                    if new_ready_time not in ORDict:
                        ORDict[new_ready_time] = []
                    ORDict[new_ready_time].append(operation)

            assigned_operations = [ORSet[row] for row, machine in zip(row_ind, col_ind) if cost_matrix[row, machine] < 999]
            unassigned_operations = [op for op in ORSet if op not in assigned_operations]

            if unassigned_operations:
                new_ready_time = min(new_availabletime) 
                if new_ready_time in ORDict.keys():
                    ORDict[new_ready_time].extend(unassigned_operations)
                else:
                    ORDict[new_ready_time] = unassigned_operations
                    
            # Process the dictionary, check flag
            if ORDict:
                ready_time   = min(ORDict)
                ORSet        = ORDict[ready_time]
               
            else: Reschedule_completion = True

        # Calculate GBest
        GBest, C_j = evaluate_LocalCost(self.d_j, C_ij, self.JSet)

        return GBest, X_ijk, S_ij, C_ij, C_j

    def LAP_LFO(self):
        ORSet            = []
        ORDict           = {} # key: ready time, value: [i, j]
        X_ijk            = np.zeros((self.I, self.J, self.K))
        S_ij             = np.zeros((self.I, self.J))
        C_ij             = np.zeros((self.I, self.J))
        dummy_JSet       = copy.deepcopy(self.JSet)
        dummy_OJSet      = copy.deepcopy(self.OJSet)

        Reschedule_completion = False
        # Start
        for j in dummy_JSet:
            ready_time = self.S_j[j]
            operation  = [dummy_OJSet[j][0], j]
            if ready_time not in ORDict:
                ORDict[ready_time] = []
            ORDict[ready_time].append(operation)

        ready_time = min(ORDict)
        ORSet      = ORDict[ready_time]

        while Reschedule_completion == False:
            # Initialize cost matrix with a high value
            num_operations = len(ORSet)
            cost_matrix    = np.full((num_operations, self.K), 999)

            # Populate the cost matrix based on constraints
            for op_idx, (i, j) in enumerate(ORSet):
                for k in range(self.K):
                    if self.h_ijk[i][j][k] == 1:
                        cost_matrix[op_idx, k] = self.S_k[k] + self.p_ijk[i][j][k]

            # Solve the assignment problem using the Hungarian algorithm
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            # Output the assignment results
            assignments = [(ORSet[row], machine) for row, machine in zip(row_ind, col_ind) if cost_matrix[row, machine] < 999]
            ORDict.pop(ready_time)
            for (i, j), k in assignments:
                X_ijk[i][j][k]      = 1
                S_ij[i][j]          = max(ready_time, self.S_k[k])
                C_ij[i][j]          = S_ij[i][j] + self.p_ijk[i][j][k]
                self.S_k[k]         = copy.deepcopy(C_ij[i][j])
                dummy_OJSet[j].remove(i) 
                if len(dummy_OJSet[j]) != 0:
                    new_ready_time = C_ij[i][j]
                    operation      = [dummy_OJSet[j][0], j]
                    if new_ready_time not in ORDict:
                        ORDict[new_ready_time] = []
                    ORDict[new_ready_time].append(operation)

            assigned_operations = [ORSet[row] for row, machine in zip(row_ind, col_ind) if cost_matrix[row, machine] < 999]
            unassigned_operations = [op for op in ORSet if op not in assigned_operations]

            if unassigned_operations:
                sorted_operation = sorted(unassigned_operations, key=lambda x: (self.n_MC_ji[x[1]][x[0]], self.d_j[x[1]], np.mean(self.p_ijk[x[0]][x[1]])))
                for i, j in sorted_operation:
                    # Select machine
                    if self.n_MC_ji[j][i] == 1:
                        k = self.MC_ji[j][i]
                        X_ijk[i][j][k] = 1
                    else:
                        mask            = np.multiply(self.p_ijk[i][j], self.h_ijk[i][j])
                        mask[mask==0]   = 999
                        sum_values      = mask + np.maximum(ready_time, self.S_k[k])
                        k               = np.argmin(sum_values)
                        X_ijk[i][j][k]  = 1  
                    # Calculate Start time, Completion time, and set new S_k
                    S_ij[i][j]          = max(ready_time, self.S_k[k])
                    C_ij[i][j]          = S_ij[i][j] + self.p_ijk[i][j][k]
                    self.S_k[k]         = copy.deepcopy(C_ij[i][j])
                    # Adjust the set and append the dictionary (if any)
                    dummy_OJSet[j].remove(i)
                    if len(dummy_OJSet[j]) != 0:
                        new_ready_time = C_ij[i][j]
                        operation      = [dummy_OJSet[j][0], j]
                        if new_ready_time not in ORDict:
                            ORDict[new_ready_time] = []
                        ORDict[new_ready_time].append(operation)

            # Process the dictionary, check flag
            if ORDict:
                ready_time   = min(ORDict)
                ORSet        = ORDict[ready_time]
            else: Reschedule_completion = True

        # Calculate GBest
        GBest, C_j = evaluate_LocalCost(self.d_j, C_ij, self.JSet)

        return GBest, X_ijk, S_ij, C_ij, C_j

    def CDR1(self):
        X_ijk            = np.zeros((self.I, self.J, self.K))
        S_ij             = np.zeros((self.I, self.J))
        C_ij             = np.zeros((self.I, self.J))
        p_mean           = np.mean(self.p_ijk*self.h_ijk, axis= 2)
        dummy_JSet       = copy.deepcopy(self.JSet)
        Reschedule_completion = False
        
        while Reschedule_completion == False:
            T_cur = np.mean(self.S_k)
            # Check if there is Tard_job
            if self.Tard_job:
                OP = self.n_j - self.n_ops_left_j
                estimated_tardiness = np.full(self.J, np.inf)
                for j in self.Tard_job:
                    estimated_tardiness[j] = T_cur + np.sum(p_mean[OP[j]:self.n_j[j], j]) - self.d_j[j]
                j = np.argmin(estimated_tardiness)
            else:
                average_slack = np.full(self.J, np.inf)
                average_slack[dummy_JSet] = (self.d_j[dummy_JSet] - T_cur)/self.n_ops_left_j[dummy_JSet]
                j = np.argmin(average_slack)

            i                   = self.n_j[j] - self.n_ops_left_j[j]
            # Find earliest available machine
            available           = np.maximum(self.S_j[j], self.S_k)
            mask                = self.h_ijk[i, j, :] == 1
            filtered_available  = available[mask]
            filtered_index      = np.argmin(filtered_available)
            k                   = np.arange(len(available))[mask][filtered_index]
            X_ijk[i][j][k]      = 1  
            # Calculate Start time, Completion time, and set new S_k
            S_ij[i][j]          = max(self.S_j[j], self.S_k[k])
            C_ij[i][j]          = S_ij[i][j] + self.p_ijk[i][j][k]
            self.S_k[k]         = copy.deepcopy(C_ij[i][j])
            # Adjust the set
            self.OJSet[j].remove(i)
            self.n_ops_left_j[j] -= 1
            if len(self.OJSet[j]) == 0:
                dummy_JSet.remove(j)
            else:
                self.S_j[j] = copy.deepcopy(C_ij[i][j])

            if not dummy_JSet:
                Reschedule_completion = True
            else:
                self.Tard_job = [j for j in dummy_JSet if self.d_j[j] < T_cur]

        # Calculate GBest
        GBest, C_j = evaluate_LocalCost(self.d_j, C_ij, self.JSet)

        return GBest, X_ijk, S_ij, C_ij, C_j

    def CDR2(self):
        X_ijk            = np.zeros((self.I, self.J, self.K))
        S_ij             = np.zeros((self.I, self.J))
        C_ij             = np.zeros((self.I, self.J))
        p_mean           = np.mean(self.p_ijk*self.h_ijk, axis= 2)
        dummy_JSet       = copy.deepcopy(self.JSet)
        Reschedule_completion = False
        print("dummy_JSet", dummy_JSet)
        while Reschedule_completion == False:
            T_cur = np.mean(self.S_k)
            OP = self.n_j - self.n_ops_left_j

            # Check if there is Tard_job
            if self.Tard_job:
                estimated_tardiness = np.full(self.J, np.inf)
                for j in self.Tard_job:
                    estimated_tardiness[j] = T_cur + np.sum(p_mean[OP[j]:self.n_j[j], j]) - self.d_j[j]
                j = np.argmin(estimated_tardiness)
            else:
                critical_ratio = np.full(self.J, np.inf)
                for j in dummy_JSet:
                    critical_ratio[j] = (self.d_j[j] - T_cur)/np.sum(p_mean[OP[j]:self.n_j[j], j])
                j = np.argmin(critical_ratio)

            i                   = self.n_j[j] - self.n_ops_left_j[j]
            # Find earliest available machine
            available           = np.maximum(self.S_j[j], self.S_k)
            mask                = self.h_ijk[i, j, :] == 1
            filtered_available  = available[mask]
            filtered_index      = np.argmin(filtered_available)
            k                   = np.arange(len(available))[mask][filtered_index]
            X_ijk[i][j][k]      = 1  
            # Calculate Start time, Completion time, and set new S_k
            S_ij[i][j]          = max(self.S_j[j], self.S_k[k])
            C_ij[i][j]          = S_ij[i][j] + self.p_ijk[i][j][k]
            self.S_k[k]         = copy.deepcopy(C_ij[i][j])
            # Adjust the set
            self.OJSet[j].remove(i)
            self.n_ops_left_j[j] -= 1
            if len(self.OJSet[j]) == 0:
                dummy_JSet.remove(j)
            else:
                self.S_j[j] = copy.deepcopy(C_ij[i][j])

            if not dummy_JSet:
                Reschedule_completion = True
            else:
                self.Tard_job = [j for j in dummy_JSet if self.d_j[j] < T_cur]

        # Calculate GBest
        GBest, C_j = evaluate_LocalCost(self.d_j, C_ij, self.JSet)

        return GBest, X_ijk, S_ij, C_ij, C_j

    def CDR4(self):
        X_ijk            = np.zeros((self.I, self.J, self.K))
        S_ij             = np.zeros((self.I, self.J))
        C_ij             = np.zeros((self.I, self.J))
        dummy_JSet       = copy.deepcopy(self.JSet)

        Reschedule_completion = False

        while Reschedule_completion == False:
            # Randomly select a job
            j                   = random.choice(dummy_JSet)
            i                   = self.n_j[j] - self.n_ops_left_j[j]
            # Find earliest available machine
            available           = np.maximum(self.S_j[j], self.S_k)
            mask                = self.h_ijk[i, j, :] == 1
            filtered_available  = available[mask]
            filtered_index      = np.argmin(filtered_available)
            k                   = np.arange(len(available))[mask][filtered_index]
            X_ijk[i][j][k]      = 1  
            # Calculate Start time, Completion time, and set new S_k
            S_ij[i][j]          = max(self.S_j[j], self.S_k[k])
            C_ij[i][j]          = S_ij[i][j] + self.p_ijk[i][j][k]
            self.S_k[k]         = copy.deepcopy(C_ij[i][j])
            # Adjust the set
            self.OJSet[j].remove(i)
            self.n_ops_left_j[j] -= 1
            if len(self.OJSet[j]) == 0:
                dummy_JSet.remove(j)
            else:
                self.S_j[j]      = copy.deepcopy(C_ij[i][j])
            if not dummy_JSet:
                Reschedule_completion = True

        # Calculate GBest
        GBest, C_j = evaluate_LocalCost(self.d_j, C_ij, self.JSet)
        return GBest, X_ijk, S_ij, C_ij, C_j


    def RouteChange_RightShift(self):
        Job_seq = copy.deepcopy(self.OJSet)
        Mch_seq = find_Mch_seq(self.K, self.X_ijk, self.C_ij, self.t)

        for breakdown_MC in self.affected_Oij.keys():
            for Oij in reversed(self.affected_Oij[breakdown_MC]):
                X_mask    =  self.X_ijk.astype(bool)
                S_ij_mask =  self.S_ij >= self.t

                indirect_interval    = []
                indirect_RouteChange = []
                cost_Modified        = []

                direct_RouteChange   = []
                i = Oij[0]
                j = Oij[1]

                ## Find start time of successor operation of Oij
                if i < int(self.n_j[j]-1):
                    S_Oij_suc = copy.deepcopy(self.S_ij[i+1][j])
                else:
                    S_Oij_suc = np.inf
                ## Find start time of soonest operation on the machine k'
                for k in self.MC_ji[j][i]:
                    if k != breakdown_MC: 
                        combined_mask = X_mask[:,:, k] & S_ij_mask
                        S_k_suc_filtered = self.S_ij[combined_mask]
                        if np.any(S_k_suc_filtered):
                            S_k_suc = np.min(S_k_suc_filtered)
                        else:
                            S_k_suc = np.inf
                        ## Check if satisfy equation
                        interval = min(S_Oij_suc, S_k_suc) - self.S_k[k]
                        
                        if self.p_ijk[i][j][k] < interval:
                            cost = self.S_k[k] + self.p_ijk[i][j][k]
                            direct_RouteChange  .append(k)
                            cost_Modified       .append(cost)
                        else:
                            indirect_RouteChange.append(k)
                            indirect_interval   .append(interval)

                ### If satisfied
                if direct_RouteChange:
                    min_cost =  min(cost_Modified)
                    k_index  =  cost_Modified.index(min_cost)
                    new_k    =  direct_RouteChange[k_index]
                    self.X_ijk[i][j][new_k] = 1
                    self.X_ijk[i][j][breakdown_MC] = 0

                    self.S_ij[i][j] = self.S_k[new_k]
                    self.C_ij[i][j] = self.S_ij[i][j] + self.p_ijk[i][j][new_k]

                ### If not
                else:
                    X_modified       = []
                    S_modified       = []
                    C_modified       = []
                    Mch_seq_modified = []
                    id_ope_onMCh     = Mch_seq[breakdown_MC].index(Oij)

                    X_RightShift, S_RightShift, C_RightShift = RightShift(breakdown_MC, id_ope_onMCh, self.S_k[breakdown_MC], Job_seq, Mch_seq, self.X_ijk, self.S_ij, self.C_ij, self.p_ijk, self.n_j)
                    cost_RightShift = evaluate_LocalCost(self.d_j, C_RightShift, self.JSet)

                    X_modified      .append(X_RightShift)
                    S_modified      .append(S_RightShift)
                    C_modified      .append(C_RightShift)
                    cost_Modified   .append(cost_RightShift)
                    Mch_seq_modified.append(Mch_seq)

                    ## Route-Change
                    max_interval =  max(indirect_interval)
                    k_index      =  indirect_interval.index(max_interval)
                    new_k        =  indirect_RouteChange[k_index]

                    dummy_X_ijk  =  copy.deepcopy(self.X_ijk)
                    dummy_X_ijk[i][j][new_k] = 1
                    dummy_X_ijk[i][j][breakdown_MC] = 0

                    ## Change Mch seq
                    dummy_Mch_seq = copy.deepcopy(Mch_seq)
                    dummy_Mch_seq[breakdown_MC].remove(Oij)
                    dummy_Mch_seq[new_k]       .insert(0, Oij)

                    ## Right-Shift
                    X_RCRS, S_RCRS, C_RCRS = RightShift(breakdown_MC, id_ope_onMCh, self.S_k[new_k], Job_seq, Mch_seq, dummy_X_ijk, self.S_ij, self.C_ij, self.p_ijk, self.n_j)
                    cost_RCRS = evaluate_LocalCost(self.d_j, C_RCRS, self.JSet)

                    X_modified      .append(X_RCRS)
                    S_modified      .append(S_RCRS)
                    C_modified      .append(C_RCRS)
                    cost_Modified   .append(cost_RCRS)
                    Mch_seq_modified.append(dummy_Mch_seq)
  
                    for new_k in self.MC_ji[j][i]:
                        # machine breakdown
                        X_mask_new_k   = self.X_ijk[:, :, new_k].astype(bool)

                        if not X_mask_new_k.any():
                            C_LastPosition = self.S_k[new_k]
                            S_Oij = self.S_j[j]
                        else:
                            C_LastPosition = self.C_ij[X_mask_new_k].max()
                            S_Oij = copy.deepcopy(C_LastPosition)
                        
                        if i < int(self.n_j[j]) - 1:
                            S_Oij_suc = copy.deepcopy(self.S_ij[i+1][j])
                        else:
                            S_Oij_suc = np.inf
                        
                        C_Oij = S_Oij + self.p_ijk[i][j][new_k]

                        dummy_X_ijk   = self.X_ijk.copy()
                        dummy_S_ij    = self.S_ij .copy()
                        dummy_C_ij    = self.C_ij .copy()
                        dummy_Mch_seq = copy.deepcopy(Mch_seq)

                        if C_Oij <= S_Oij_suc:
                            dummy_X_ijk[i][j][breakdown_MC] = 0
                            dummy_X_ijk[i][j][new_k] = 1
                            dummy_S_ij[i][j] = copy.deepcopy(S_Oij)
                            dummy_C_ij[i][j] = copy.deepcopy(C_Oij)
                            cost_LastPos = evaluate_LocalCost(self.d_j, dummy_C_ij, self.JSet)
                            
                            dummy_Mch_seq[breakdown_MC].remove(Oij)
                            dummy_Mch_seq[new_k]       .append(Oij)
                        else:
                            cost_LastPos = np.inf
                        
                        X_modified      .append(dummy_X_ijk)
                        S_modified      .append(dummy_S_ij)
                        C_modified      .append(dummy_C_ij)
                        cost_Modified   .append(cost_LastPos)
                        Mch_seq_modified.append(dummy_Mch_seq)

                    min_cost      =  min(cost_Modified)
                    sched_id      =  cost_Modified.index(min_cost)
                    self.X_ijk    =  X_modified      [sched_id]
                    self.S_ij     =  S_modified      [sched_id]
                    self.C_ij     =  C_modified      [sched_id]
                    Mch_seq       =  Mch_seq_modified[sched_id]

        GBest, C_j = evaluate_LocalCost(self.d_j, self.C_ij, self.JSet)

        return GBest, self.X_ijk, self.S_ij, self.C_ij, C_j