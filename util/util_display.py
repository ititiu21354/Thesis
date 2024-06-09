import pandas as pd
import copy

import datetime
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

def plot(J, K, n_j, X_ijk, S_ij, C_ij, MB_record, t):
    gantt_data = []
    start_date = datetime.datetime(year=2024, month=7, day=1, hour=8, minute=0, second=0)
    machines_with_jobs = set()

    # Generate a colormap with a sufficient number of colors
    cmap = plt.get_cmap('tab20c', J + 1)
    colors = [mcolors.rgb2hex(cmap(i)) for i in np.linspace(0, 1, J + 1)]
    
    # Create a color dictionary mapping resources to their colors
    color_dict = {f'Job {j}': colors[j % len(colors)] for j in range(J)}
    color_dict['No Job'] = '#e6ecf5'
    
    for k in range(K):
        # Draw operations
        for j in range(J):
            job_color = colors[j % len(colors)]
            for i in range(int(n_j[j])):
                if X_ijk[i][j][k] == 1:
                    # Convert start and completion times to datetime objects
                    start_time = start_date + datetime.timedelta(seconds=S_ij[i][j])
                    completion_time = start_date + datetime.timedelta(seconds=C_ij[i][j])
                    
                    gantt_data.append(dict(Task=f'Machine {k}', Start=start_time, Finish=completion_time,
                                           Resource=f'Job {j}'))
                    machines_with_jobs.add(k)

        if k not in machines_with_jobs:
            dummy_start_time = start_date
            dummy_end_time = start_date + datetime.timedelta(seconds=1)  # Minimal duration to display the task
            gantt_data.append(dict(Task=f'Machine {k}', Start=dummy_start_time, Finish=dummy_end_time,
                                   Resource='No Job'))
        
    

    # Creating the Gantt chart figure
    fig = ff.create_gantt(gantt_data, 
                          index_col='Resource', 
                          show_colorbar=True,
                          group_tasks=True, 
                          showgrid_x=True, 
                          showgrid_y=True
                          ,colors=color_dict
                          )
   
    for k in range(K):
        """Draw machine breakdown"""
        if k in MB_record:
            for MB_starttime, MB_endtime in MB_record[k]:
                t1_datetime = start_date + datetime.timedelta(seconds=MB_starttime)
                t2_datetime = start_date + datetime.timedelta(seconds=MB_endtime)
                # Add rectangle for the machine breakdown
                gantt_data.append(dict(Task=f'Machine {k}', Start=t1_datetime, Finish=t2_datetime,
                                    Resource=f'Machine Breakdown', Color='white'))
                # Add "X" spanning the rectangle as annotation
                fig.add_shape(
                    type="rect",
                    x0=t1_datetime,
                    y0=K-k-1 - 0.18,  # Adjust for the vertical position of the machine
                    x1=t2_datetime,
                    y1=K-k-1 + 0.18,
                    line=dict(color="black"),
                    fillcolor="white",
                    opacity=1)
                fig.add_annotation(
                    x=(t1_datetime + (t2_datetime - t1_datetime) / 2),
                    y=K-k-1,
                    text="X",
                    showarrow=False,
                    font=dict(size=10, color="black"),
                    align="center",
                    valign="middle")
    
    # Rescheduling point
    vertical_line_time = start_date + datetime.timedelta(seconds=t)
    fig.add_shape(
        type="line",
        x0=vertical_line_time,
        y0=-0.5,
        x1=vertical_line_time,
        y1=K - 0.5,
        line=dict(color="black", dash="dash")
    )
    # Ensure the legend is sorted in ascending order
    fig.update_layout(
        title='Gantt Chart',
        xaxis=dict(title='Time', tickformat='%H:%M'),
        yaxis=dict(title='Machine'),
        legend=dict(traceorder="normal",)
        )

    return fig

# def plot(J, K, n_j, X_ijk, S_ij, C_ij):
#     gantt_data = []
#     colors = ['#FFA07A', '#98FB98', '#87CEFA', '#FFC0CB', '#FFFFE0', '#00FFFF', '#FFE4E1', '#F0E68C']
#     start_date = datetime.datetime(year=2024, month=6, day=1, hour=8, minute=0)
#     for k in range(K):
#         for j in range(J):
#             for i in range(int(n_j[j])):
#                 if X_ijk[i][j][k] == 1:
#                     job_color = colors[j % len(colors)]
                    
#                     # Convert start and completion times to datetime objects
#                     start_time      = start_date + datetime.timedelta(minutes=S_ij[i][j])
#                     completion_time = start_date + datetime.timedelta(minutes=C_ij[i][j])
                    
#                     gantt_data.append(dict(Task=f'Machine {k}', Start=start_time, Finish=completion_time,
#                                         Resource=f'Job {j}', Color=job_color))

#     # Creating the Gantt chart figure
#     fig = ff.create_gantt(gantt_data, index_col='Resource', show_colorbar=True,
#                         group_tasks=True, showgrid_x=True, showgrid_y=True)

#     fig.update_layout(title='Gantt Chart',
#                     xaxis=dict(title='Time', tickformat='%H:%M'),  # Set x-axis tick format to hour:minute
#                     yaxis=dict(title='Machine'))

#     return fig

def pretty_table(J, I, n_j, X_ijk, S_ij, C_ij):
    Schedule_M          = np.full((J, I), -999)
    Schedule_S          = np.full((J, I), -999)
    Schedule_C          = np.full((J, I), -999)

    for j in range(J):
        for i in range(int(n_j[j])):
            Schedule_M[j][i] = np.argmax(X_ijk[i][j])
            Schedule_S[j][i] = copy.deepcopy(S_ij[i][j])
            Schedule_C[j][i] = copy.deepcopy(C_ij[i][j])

    # Change to int64
    Schedule_M = Schedule_M.astype(np.int64)
    Schedule_S = Schedule_S.astype(np.int64)
    Schedule_C = Schedule_C.astype(np.int64)

    # Reshape the arrays to convert them into 1D arrays
    M_1d = Schedule_M.reshape(-1)
    S_1d = Schedule_S.reshape(-1)
    C_1d = Schedule_C.reshape(-1)

    # Create index arrays for i and j
    j_index = np.repeat(np.arange(J), I)
    i_index = np.tile(np.arange(I), J)

    # Create a DataFrame
    data = {
        'Job'       : j_index,
        'Ope'       : i_index,
        'MC'        : M_1d,
        'Start'     : S_1d,
        'End'       : C_1d
    }

    df = pd.DataFrame(data)
    
    return df


def summarise(OSet, Title):    
    Set = list(range(len(OSet)))
    df  = pd.DataFrame({'Job': Set, 'Operations': OSet})
    df['Operations'] = df['Operations'].apply(lambda x: '-'.join(map(str, x)) if x else 'None')

    print(Title)
    return df