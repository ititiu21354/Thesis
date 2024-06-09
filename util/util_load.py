import xlwings as xw
import pandas as pd
import numpy as np


def load(defined_name):
    wb                 = xw.Book(r"D:\D\University\4th year\Thesis\Code T4\Small_example.xlsx")
    named_range        = wb.names[defined_name]
    data               = named_range.refers_to_range.value
    df                 = pd.DataFrame(data)
    return df


def machine_data(J, I, K, n_j, MachineCapability):
    MC_ji              = [[[] for _ in range(I)] for _ in range(J)]
    n_MC_ji            = [[[] for _ in range(I)] for _ in range(J)]
    h_ijk              = np.zeros((I, J, K))

    for _, row in MachineCapability.iterrows():

        j              = row['Job']
        i              = row['Operation']
        machines       = row['Machine capable']

        machines_list  = [int(machine.strip()) for machine in machines.split(',') if machine.strip()]
        MC_ji   [j][i] = machines_list
        n_MC_ji [j][i] = int(len(machines_list))
    
    for j in range(J):
        for i in range(int(n_j[j])):
            for k in range(K):
                if k in MC_ji[j][i]:
                    h_ijk[i][j][k] = 1

    return MC_ji, n_MC_ji, h_ijk


def read_txt(filename):
    # Read data from the text file
    with open(filename, "r") as file:
        data = file.readlines()

    # Extracting parameters J, K, I
    J = int(data[0].strip())  # Number of jobs
    K = int(data[1].strip())  # Number of machines
    I = max(map(int, data[2].strip().split()))  # Maximum number of operations per job
    d_j = np.array(list(map(int, data[3].strip().split())), dtype=int)

    # Initialize p_ijk matrix, n_j vector, and MC_ji list of lists
    p_ijk = np.full((I, J, K), 999)
    h_ijk = np.zeros((I, J, K), dtype=int)

    n_j     = np.zeros(J, dtype=int)
    MC_ji   = [[[] for _ in range(I)] for _ in range(J)]
    n_MC_ji = [[0 for _ in range(I)] for _ in range(J)]

    # Extract job, operation, machine, and processing time data
    for line in data[4:]:
        job, operation, machine, processingtime = map(int, line.split())
        job       = int(job)       - 1     # Adjust to 0-based indexing
        operation = int(operation) - 1     # Adjust to 0-based indexing
        machine   = int(machine)   - 1     # Adjust to 0-based indexing
        # Update p_ijk and h_ijk matrix
        p_ijk[operation, job, machine] = processingtime
        h_ijk[operation, job, machine] = 1

        # Update n_j vector
        n_j[job] = max(n_j[job], operation + 1)

        # Update MC_ji list
        MC_ji[job][operation].append(machine)
        n_MC_ji[job][operation] += 1
    
    OperationPool = pd.DataFrame({
        "Job": np.arange(J), 
        "Num operation left": n_j  
    })

    return J, I, K, p_ijk, h_ijk, d_j, n_j, MC_ji, n_MC_ji, OperationPool


def insert_duedate(filename):

    # Read data from the text file
    with open(filename, "r") as file:
        data = file.readlines()

    # Extracting parameters J, K, I
    J = int(data[0].strip())  # Number of jobs
    K = int(data[1].strip())  # Number of machines
    I = max(int(line.split()[1]) for line in data[3:])  # Maximum number of operations per job

    # Initialize p_ijk matrix, n_j vector, and MC_ji list of lists
    p_ijk = np.full((I, J, K), 999)
    h_ijk = np.zeros((I, J, K), dtype=int)

    n_j   = np.zeros(J, dtype=int)
    MC_ji = [[[] for _ in range(I)] for _ in range(J)]

    # Extract job, operation, machine, and processing time data
    for line in data[3:]:
        job, operation, machine, processingtime = map(int, line.split())
        job       -= 1      # Adjust to 0-based indexing
        operation -= 1      # Adjust to 0-based indexing
        machine   -= 1      # Adjust to 0-based indexing

        # Update p_ijk and h_ijk matrix
        p_ijk[operation, job, machine] = processingtime
        h_ijk[operation, job, machine] = 1

        # Update n_j vector
        n_j[job] = max(n_j[job], operation + 1)

        # Update MC_ji list
        if machine not in MC_ji[job][operation]:
            MC_ji[job][operation].append(machine)

    weighted_processing_times = p_ijk * h_ijk
    valid_machines            = np.sum(h_ijk, axis=2)
    p_ij                      = np.divide(weighted_processing_times.sum(axis=2), valid_machines, where=(valid_machines!=0))
    p_ij[valid_machines == 0] = 0
    d_j                       = np.round((np.random.uniform(50, 150, size=J) * np.sum(p_ij, axis=0)),0).astype(int)

    d_j_str = ' '.join(str(int(value)) for value in d_j)

    # Insert the new line after line 3 (index 2)
    data.insert(3, d_j_str + '\n')

    # Write the modified content back to the file
    with open(filename, 'w') as file:
        file.writelines(data)

    return