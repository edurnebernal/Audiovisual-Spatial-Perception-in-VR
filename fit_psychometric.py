import utils
import numpy as np
import psignifit as ps
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
mpl.rcParams.update(mpl.rcParamsDefault)

##############################################################################################################
# Select the experiment to analyze and obtain pyshicometric curves
##############################################################################################################
EXP = 0
OUTLIERS_PATH = './data/outlier_points.csv'
DATA_PATH = './data'
##############################################################################################################

DICT = {0: 'stationary', 1: 'pursuit', 2: 'reorientation_slow', 3:'reorientation_fast'}
DIR = {0: [0, 0], 1: [1, -1], 2: [1, -1], 3: [1, -1]}
TITLES = {0: ['Stationary','Stationary'], 
          1: ['PursuitR','PursuitL'], 
          2: ['Re-orientation25R','Re-orientation25L'], 
          3: ['Re-orientation50R','Re-orientation50L']}

data, _ = utils.load_and_filter_eyetracking_data(DATA_PATH + '\data_' + DICT[EXP], cvs_files_path=DATA_PATH, file_name=DICT[EXP], base=DICT[EXP]=='stationary')
ids =  [400,401,402,403,405,406,408,409,410,411,412,413,414,415,416,417,418,419,420,421]
data = data[data['ID'].isin(ids)]

# Read csv with outliers
outliers = pd.read_csv(OUTLIERS_PATH)
outliers = outliers[outliers['EXP'] == EXP]

# Remove outliers
for i in range(len(outliers)):
    data = data.drop(data[(data['ID'] == outliers['p'].values[i]) & (data['event'] == outliers['trial'].values[i])].index)


options = {'sigmoidName': 'logistic', 
           'useGPU': 1,
           'expType': 'equalAsymptote',
           'fixedParameters': 0,
           'instantPlot': False}

options['fixedPars'] = np.ones([5,1])*np.nan


results_d = ps.psignifit(utils.parse_data(data,DIR[EXP][0]), options)
results_l = ps.psignifit(utils.parse_data(data,DIR[EXP][1]), options)


# Print PSE and DT25 and DT75 values for right direction
print('Experiment:', DICT[EXP], ' Right direction')
print('----------------------------------------')
print('PSE:', ps.getThreshold(results_d,0.5)[0])
print('DT25:', ps.getThreshold(results_d,0.25)[0])
print('DT75:', ps.getThreshold(results_d,0.75)[0])

utils.plotPsych(results_d, lineColor=mpl.colors.to_rgba('dodgerblue', alpha=None),
                        dataColor=mpl.colors.to_rgba('dodgerblue', alpha=None),
                        showse = True,
                        se = utils.standard_error(data,DIR[EXP][0]),
                        colorse = mpl.colors.to_rgba('dodgerblue', alpha=0.2),
                        CIthresh=True,
                        DTarea = False,
                        xLabel = 'Offset (deg)',
                        yLabel = 'Probability of right answer',
                        title = TITLES[EXP][0])



utils.plotPsych(results_l, lineColor=mpl.colors.to_rgba('darkslateblue', alpha=None),
                        dataColor=mpl.colors.to_rgba('slateblue', alpha=None),
                        showse = True,
                        se = utils.standard_error(data,DIR[EXP][1]),
                        colorse = mpl.colors.to_rgba('slateblue', alpha=0.2),
                        CIthresh=True,
                        DTarea = True,
                        xLabel = 'Offset (deg)',
                        yLabel = 'Probability of right answer',
                        title = TITLES[EXP][1])

# Print PSE and DT25 and DT75 values for right direction
print('Experiment:', DICT[EXP], ' Left direction')
print('----------------------------------------')
print('PSE:', ps.getThreshold(results_l,0.5)[0])
print('DT25:', ps.getThreshold(results_l,0.25)[0])
print('DT75:', ps.getThreshold(results_l,0.75)[0])
plt.show()