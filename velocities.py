import os
import numpy as np
import pandas as pd
import utils


############################################################################################
# Plot the head velocities for the different experiments and obtain the trials with 
# oulier velicities
############################################################################################
DATA_PATH = './data'
mean_vel = {0: [], 1: [], 2: [], 3: []}
DICT = {0: 'stationary', 1: 'pursuit', 2: 'reorientation_slow', 3:'reorientation_fast'}
DIR = {0: [0, 0], 1: [1, -1], 2: [1, -1], 3: [1, -1]}
OUTPUT_FILE = './data/outlier_points.csv'
trial_id = {0: [], 1: [], 2: [], 3: []}
Time = {0: [500], 1: [800], 2: [2600], 3: [800]} # Time to remove from the begining and end of the trial (ms), since the stimulus is not present
#############################################################################################

for EXP in range(4):

    data, eye_track = utils.load_and_filter_eyetracking_data(DATA_PATH + '\data_' + DICT[EXP], cvs_files_path=DATA_PATH, file_name=DICT[EXP], base=DICT[EXP]=='stationary')

    # Compute the mean velocity for each trial
    for p in eye_track['ID'].unique():
        p_data = eye_track[eye_track['ID'] == p]
        
        for trial in p_data['event'].unique():
            # Get the data for the trial
            trial_et = p_data[p_data['event'] == trial]

            # Remove the first and last ms of the trial
            trial_et = trial_et[trial_et['timestamp'] > Time[EXP][0]]
            trial_et = trial_et[trial_et['timestamp'] < trial_et['timestamp'].max() - Time[EXP][0]]

            if len(trial_et) < 2:
                continue
            u_head = trial_et['u_head'].values * 360
            t = trial_et['timestamp'].values

            if np.abs(u_head[-1] - u_head[0]) < 50 and EXP != 0:
                continue

            # Compute the horizontal velocity of the head as u2 -u1 / t2 - t1
            vel = np.abs(u_head[-1] - u_head[0]) / ((t[-1] - t[0])/1000)

            # Save the velocity
            mean_vel[EXP].append(vel)
            trial_id[EXP].append([trial, p])

# Write the outlier points in a csv file with columns EXP, p, trial
df = pd.DataFrame(columns=['EXP', 'p', 'trial'])
for EXP in range(4):
    Q1 = np.quantile(mean_vel[EXP], 0.25)
    Q3 = np.quantile(mean_vel[EXP], 0.75)
    IQR = Q3 - Q1
    for i in range(len(trial_id[EXP])):
        if mean_vel[EXP][i] < Q1 - 1.5 * IQR or mean_vel[EXP][i] > Q3 + 1.5 * IQR:
            df = df.append({'EXP': EXP, 'p': trial_id[EXP][i][1], 'trial': trial_id[EXP][i][0]}, ignore_index=True)

df.to_csv(OUTPUT_FILE, index=False)

# print th Q1 and Q3 for each condition
for EXP in range(4):
    Q1 = np.quantile(mean_vel[EXP], 0.25)
    Q3 = np.quantile(mean_vel[EXP], 0.75)
    mean = np.mean(mean_vel[EXP])
    print('Q1 for ' + DICT[EXP] + ': ' + str(Q1))
    print('Q3 for ' + DICT[EXP] + ': ' + str(Q3))
    print('IQR for ' + DICT[EXP] + ': ' + str(Q3 - Q1))
    print('Upper bound for ' + DICT[EXP] + ': ' + str(Q3 + 1.5 * (Q3 - Q1)))
    print('Lower bound for ' + DICT[EXP] + ': ' + str(Q1 - 1.5 * (Q3 - Q1)))
    print('Mean for ' + DICT[EXP] + ': ' + str(mean))
    print('')







