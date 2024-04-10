import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def load_data_app(file_path):

    participants_folder_data = os.listdir(file_path)
    
    data = pd.DataFrame(columns=['ID', 'event', 'enemy','rot_time','answer','offset'])
 
    for nump, p_fd in enumerate(participants_folder_data):
        # Chek if path string corresponds to a number character
        if not p_fd.isdigit():
            continue

        # Read the cvs file as int
        participant_data = pd.read_csv(os.path.join(file_path, p_fd, p_fd + '.csv'))

        # Add the participant ID to the data in the first column
        participant_data.insert(0, 'ID', int(p_fd))
        # Change the column names
        participant_data.columns = ['ID', 'event', 'enemy','rot_time','answer','offset']
    
        # remove test trials
        participant_data = participant_data[participant_data['event'] >= 0]

        data = data.append(participant_data, ignore_index=True)

    return data

############################################################################################
# Plot the data for the western grab gun experiment (accuracy and confusion matrix)
############################################################################################

STIMULI_ORDER = [0,1,2,3]
DATA_PATH = 'E:\[AV-Binding]\Public repository\data\data_application'
data_file = load_data_app(DATA_PATH)


# For each participant compute the accuracy (enemy == answer) when offset present or not
acc_off = []
acc_n_off = []
improvement = []

for p in data_file['ID'].unique():
    # Get the data for the participant
    p_data = data_file[data_file['ID'] == p]
    # Compute the accuracy for the trials with offset
    acc_off.append(np.mean(p_data[p_data['offset'] == True]['enemy'] == p_data[p_data['offset'] == True]['answer']))
    # Compute the accuracy for the trials without offset
    acc_n_off.append(np.mean(p_data[p_data['offset'] == False]['enemy'] == p_data[p_data['offset'] == False]['answer']))
    # Compute the improvement
    improvement.append(acc_off[-1] - acc_n_off[-1])

# Plot acc_off y acc_n_off as boxplots using seaborn using different colors
sns.boxplot(data=[acc_n_off, acc_off], palette={0: '#ee8866', 1: '#bbcc33'}, orient='v')
sns.swarmplot(data=[acc_n_off, acc_off], palette={0: '#ee8866', 1: '#bbcc33'}, size=10, orient='v', linewidth = 1, alpha=0.4)
# Add the mean as a plot point connected by a line for each boxplot with the value of the mean
plt.plot([0], [np.median(acc_n_off)], color='grey', alpha=0.4, marker='o', markersize=10, markerfacecolor='white', markeredgecolor='grey')
plt.plot([1], [np.median(acc_off)], color='grey', alpha=0.4, marker='o', markersize=10, markerfacecolor='white', markeredgecolor='grey')
# Plot the exact value of the mean
plt.text(0, np.median(acc_n_off), str(round(np.median(acc_n_off), 2)), horizontalalignment='center', verticalalignment='bottom')
plt.text(1, np.median(acc_off), str(round(np.median(acc_off), 2)), horizontalalignment='center', verticalalignment='bottom')

plt.xticks([0,1], ['No Offset', 'Offset'])
plt.ylabel('Accuracy')
plt.title('Accuracy with and without offset')
# Set y range to 0 - 1
plt.ylim(0,1)
plt.show()

# Compute the distribution of answers across the 4 stimuli when the offset is present or not for each enemy
# Create a dataframe to store the data
data = pd.DataFrame(columns=['enemy', 'stimulus', 'offset', 'answer'])

for e in data_file['enemy'].unique():
    # Get the data for the enemy
    e_data = data_file[data_file['enemy'] == e]
    # For each stimulus compute the distribution of answers when offset is present or not
    for s in STIMULI_ORDER:
        # Compute the distribution of answers when offset is present or not
        offset = np.mean(e_data[e_data['offset'] == True]['answer'] == s)
        n_offset = np.mean(e_data[e_data['offset'] == False]['answer'] == s)
        # Add the data to the dataframe
        data = data.append({'enemy': e, 'stimulus': s, 'offset': True, 'answer': offset}, ignore_index=True)
        data = data.append({'enemy': e, 'stimulus': s, 'offset': False, 'answer': n_offset}, ignore_index=True)

# Plot a confusion matrix for each enemy versus the 4 stimuli when offset is present or not
import seaborn as sns

# Plot the confusion matrix for the no  offset condition

data_n_off = data[data['offset'] == False]
i = data_n_off['enemy']
j = data_n_off['stimulus']
z = data_n_off['answer'].values

# Create a dataframe with the data
df = pd.DataFrame(z, index=[i, j]).unstack()

palette = sns.cubehelix_palette(start=.3, rot=-.75, as_cmap=True)

# Plot the confusion matrix
sns.set(font_scale=1.5)
sns.heatmap(df, annot=True, cmap=palette, cbar=False, square=True, fmt='.2f', vmin=0, vmax=1)
plt.title('Audio aligned with the visual stimulus (offset=0º)')
plt.xticks([0.5, 1.5, 2.5, 3.5], ['Stimulus 1', 'Stimulus 2', 'Stimulus 3', 'Stimulus 4'])
plt.yticks([0.5, 1.5, 2.5, 3.5], ['Stimulus 1', 'Stimulus 2', 'Stimulus 3', 'Stimulus 4'])
plt.xlabel('Target chosen by the participants')
plt.ylabel('Audio-visual stimuli (enemy)')
plt.show()

# Plot the confusion matrix for the no  offset condition

data_n_off = data[data['offset'] == True]
i = data_n_off['enemy']
j = data_n_off['stimulus']
z = data_n_off['answer'].values

# Create a dataframe with the data
df = pd.DataFrame(z, index=[i, j]).unstack()
# Plot the confusion matrix
sns.set(font_scale=1.5)
sns.heatmap(df, annot=True, cmap=palette, cbar=False, square=True, fmt='.2f', vmin=0, vmax=1)
plt.title('Audio with a spatial offset of 13.54º to the left')
plt.xlabel('Target chosen by the participants')
# Set x and y labels as Stimulus 1, Stimulus 2, Stimulus 3 and Stimulus 4
plt.xticks([0.5, 1.5, 2.5, 3.5], ['Stimulus 1', 'Stimulus 2', 'Stimulus 3', 'Stimulus 4'])
plt.yticks([0.5, 1.5, 2.5, 3.5], ['Stimulus 1', 'Stimulus 2', 'Stimulus 3', 'Stimulus 4'])
plt.ylabel('Audio-visual stimuli (enemy)')
plt.show()



