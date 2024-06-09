import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import random
import xlwings as xw
import time
from util.util_action import evaluate_LocalCost

def select_machine(random_number, operation, job, n_MC_ji, MC_ji):
    section_probability      = 1 / n_MC_ji[job][operation]
    cumulative_probabilities = [section_probability * i for i in range(1, int(n_MC_ji[job][operation]) + 1)]
    mc_selection             = False
    for i, probability in enumerate(cumulative_probabilities):
        if random_number <= probability:
            mc_selection = MC_ji[job][operation][i]
            break 
    return mc_selection


def random_solution(chromosome_len, A, N):
    population = []
    for _ in range(N):
        OA    = np.random.permutation(A)
        MS    = np.round(np.random.random(chromosome_len),2)
        population.append((OA,MS))
    return population


def decoding(S_k, S_j, OA, MS, chromosome_len, n_MC_ji, MC_ji, I, J, K, JSet, p_ijk, d_j, n_j, n_ops_left_j):
    S_ij      = np.zeros((I, J))
    C_ij      = np.zeros((I, J))
    X_ijk     = np.zeros((I, J, K))

    Sol       = []
    counts    = {}
    for gene in range(chromosome_len):
        # Job
        job = int(OA[gene])
        # Operation
        if job not in counts:
            counts[job] = -1 + n_j[job] - n_ops_left_j[job]
        counts[job] += 1
        operation   = type(job)(counts[job])
        # Machine
        random_num = MS[gene]
        machine = select_machine(random_num, operation, job, n_MC_ji, MC_ji)
        # Add to solution
        sublist = [operation, job, machine]
        Sol.append(sublist)
    
    # X_ij
    for i, j, k in Sol:
        X_ijk[i][j][k] = 1
    
    # S_ij and C_ij
    job_seq     = [[] for _ in range(J)]
    machine_seq = [[] for _ in range(K)]

    for i, j, k in Sol:
        if len(job_seq[j]) == 0 and len(machine_seq[k]) == 0:
            S_ij[i][j] = max(S_k[k], S_j[j])
        else: 
            if len(job_seq[j]) == 0: 
                pre_ope_in_job = S_j[j]
            else: 
                [a,b] = job_seq[j][-1]
                pre_ope_in_job = C_ij[a][b]
            if len(machine_seq[k]) == 0: 
                pre_ope_in_machine = S_k[k]
            else:
                [c,d] = machine_seq      [k][-1]
                pre_ope_in_machine = C_ij[c][d]
            
            S_ij[i][j] = max(pre_ope_in_job, pre_ope_in_machine)

        scheduled_Oij = [i, j] 
        job_seq     [j].append(scheduled_Oij)
        machine_seq [k].append(scheduled_Oij)    
        
        C_ij[i][j] = S_ij[i][j] + X_ijk[i][j][k] * p_ijk[i][j][k]

    sum_tardiness, C_j = evaluate_LocalCost(d_j, C_ij, JSet)

    return sum_tardiness, X_ijk, S_ij, C_ij, C_j


def evaluate(population, indices, chromosome_len, fitness, n_MC_ji, MC_ji, I, J, K, JSet, p_ijk, d_j, n_j, n_ops_left_j, S_k, S_j):
    for n in indices:
        OA                                    = population[n][0]
        MS                                    = population[n][1]
        sum_tardiness, X_ijk, S_ij, C_ij, C_j = decoding(S_k, S_j, OA, MS, chromosome_len, n_MC_ji, MC_ji, I, J, K, JSet, p_ijk, d_j, n_j, n_ops_left_j)
        fitness [n]                           = sum_tardiness
    
    return fitness


def tournament_selection(fitness, N, tournament_size):
    target_length = N//2
    selected_indices = []
    for _ in range(target_length):
        participants = np.random.choice(range(N), size=tournament_size, replace=False)
        best_idx = participants[0]
        for idx in participants[1:]:
            if fitness[idx] < fitness[best_idx]:
                best_idx = idx
        
        selected_indices.append(best_idx)

    # Assuming the rest are unselected
    unselected_indices = np.setdiff1d(range(N), selected_indices).tolist()
    if len(unselected_indices) > target_length:
        unselected_indices[:] = unselected_indices[:target_length]

    return selected_indices, unselected_indices


def crossover(population, parent_indices, child_indices, crossover_rate, chromosome_len):
    unreplaced_indices = child_indices.copy()
    while len(unreplaced_indices) >=2:
        chosen_parents_indices    = random.sample(parent_indices, 2)
        parent1                   = population[chosen_parents_indices[0]]
        parent2                   = population[chosen_parents_indices[1]]
        if random.random() <= crossover_rate:
            offspring1 = parent1[0].copy(), parent1[1].copy()
            offspring2 = parent2[0].copy(), parent2[1].copy()

            point = random.randint(1, chromosome_len - 1)

            # Perform crossover for each chromosome
            for chromosome in range(2):
                offspring1[chromosome][:point] = parent1[chromosome][:point].copy()
                offspring1[chromosome][point:] = parent2[chromosome][point:].copy()
                
                offspring2[chromosome][:point] = parent2[chromosome][:point].copy()
                offspring2[chromosome][point:] = parent1[chromosome][point:].copy()

            child1_index              = unreplaced_indices.pop(0)
            child2_index              = unreplaced_indices.pop(0)

            population[child1_index]  = offspring1
            population[child2_index]  = offspring2
    
    return population


def mutate(population, child_indices, mutation_rate, chromosome_len, J):
    for child_index in child_indices:
        child = population[child_index]

        # Mutate OA
        OA = child[0]
        for i in range(chromosome_len):
            if random.random() < mutation_rate:
                # Mutate by randomly selecting a new integer value within the range J
                OA[i] = random.randint(0, J-1)

        # Mutate MS
        MS = child[1]
        for i in range(chromosome_len):
            if random.random() < mutation_rate:
                # Mutate by generating a new random number
                MS[i] = round(random.random(),2)

        # Update the mutated child in the population
        population[child_index] = child

    return population


def correction_offsprings(population, child_indices, value_counts, chromosome_len):
    for n in child_indices:
        chromosomes         = population[n][0]
        value_counts_copy   = value_counts .copy()

        for i in range(chromosome_len):
            value_counts_copy[chromosomes[i]] -= 1
            if value_counts_copy[chromosomes[i]] < 0:
                for value in value_counts_copy.keys():
                    if value_counts_copy[value] > 0 and value != chromosomes[i]:
                        chromosomes[i] = value
                        value_counts_copy[value] -= 1
                        break
    return population


def GeneticAlgorithm (S_k, S_j, JSet, OJSet, J, I, K, 
                      p_ijk, h_ijk, d_j, n_j, n_ops_left_j, 
                      MC_ji, n_MC_ji, OperationPool, maxtime):
    
    N                   = 50
    max_Generation      = 50
    max_No_improve      = 10
    start_time          = time.time()
    GBest               = float('inf')
    crossover_rate      = 0.7
    mutation_rate       = 0.3
    tournament_size     = 3

    fitness             = np.zeros((N))
    A                   = np.repeat(OperationPool['Job'].values, OperationPool['Num operation left'].values)
    chromosome_len      = len(A)

    if chromosome_len >= 2:
        value_counts    = dict(zip(OperationPool['Job'], OperationPool['Num operation left']))

        # Main model
        population                          = random_solution        (chromosome_len, A, N)
        fitness                             = evaluate               (population, range(N), chromosome_len, fitness, 
                                                                      n_MC_ji, MC_ji, I, J, K, JSet, p_ijk, d_j, n_j, n_ops_left_j, S_k, S_j)
        generation                          = 0
        no_improve                          = 0
        elapsed_time                        = 0
        while generation < max_Generation and no_improve < max_No_improve and elapsed_time < maxtime:
            parent_indices, child_indices   = tournament_selection   (fitness, N, tournament_size)
            population                      = crossover              (population, parent_indices, child_indices, crossover_rate, chromosome_len)
            population                      = mutate                 (population, child_indices, mutation_rate, chromosome_len, J)
            population                      = correction_offsprings  (population, child_indices, value_counts, chromosome_len)
            fitness                         = evaluate               (population, child_indices, chromosome_len, fitness, 
                                                                      n_MC_ji, MC_ji, I, J, K, JSet, p_ijk, d_j, n_j, n_ops_left_j, S_k, S_j)
            
            LBest = np.min(fitness)
            
            if LBest < GBest:
                LB_id = np.argmin(fitness)
                GBest      = LBest.copy()
                G_OA       = population[LB_id][0].copy()
                G_MS       = population[LB_id][1].copy()
                no_improve = 0
            else:
                no_improve += 1


            generation += 1
            elapsed_time = time.time() - start_time
    
        sum_tardiness, X_ijk, S_ij, C_ij, C_j = decoding(S_k, S_j, G_OA, G_MS, chromosome_len, 
                                                    n_MC_ji, MC_ji, I, J, K, JSet, p_ijk, d_j, n_j, n_ops_left_j)

    else:
        X_ijk = np.zeros((I, J, K))
        S_ij  = np.zeros((I, J))
        C_ij  = np.zeros((I, J))

        for j in JSet:
            for i in OJSet[j]:
                mask            = np.multiply(p_ijk[i][j], h_ijk[i][j])
                mask[mask==0]   = np.inf
                sum_values      = mask + S_k
                k               = np.argmin(sum_values)
                X_ijk[i][j][k]  = 1 
                S_ij[i][j]      = max(S_k[k], S_j[j])
                C_ij[i][j]      = S_ij[i][j] + p_ijk[i][j][k]
            
        GBest, C_j = evaluate_LocalCost(d_j, C_ij, JSet)

    return GBest, X_ijk, S_ij, C_ij, C_j