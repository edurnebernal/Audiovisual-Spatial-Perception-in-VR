import utils
import numpy as np
import psignifit as ps
import matplotlib as mpl
import pandas as pd
import tqdm
mpl.rcParams.update(mpl.rcParamsDefault)

############################################################################################################
# Script to compute the individual PSE and DT25 and DT75 values for each participant
############################################################################################################
OPTIONS = {'sigmoidName': 'logistic', 
           'useGPU': 1,
           'expType': 'equalAsymptote',
           'fixedParameters': 0,
           'instantPlot': False}

OPTIONS['fixedPars'] = np.ones([5,1])*np.nan
############################################################################################################
DATA_PATH = './data'
OUTLIERS_FILE = './data/outlier_points.csv'
OUTPUT_FILE = './data/pse_dts.csv'
EXPERIMENTS_DATA = {}
KEYs = ['vel0', 'pursuitL', 'pursuitR', 'vel25L', 'vel25R', 'vel50L', 'vel50R']
DICT = {0: 'stationary', 1: 'pursuit', 2: 'reorientation_slow', 3:'reorientation_fast'}
DIR = {'vel0': 0, 'pursuitL': -1, 'pursuitR': 1, 'vel25L': -1, 'vel25R': 1, 'vel50L': -1, 'vel50R': 1}
ids = [400,401,402,403,405,406,408,409,410,411,412,413,414,415,416,417,418,419,420,421]


n_exp = 0
for EXP in [0, 1, 2, 3]:
    data, _ = utils.load_and_filter_eyetracking_data(DATA_PATH + '\data_' + DICT[EXP], cvs_files_path=DATA_PATH, file_name=DICT[EXP], base=DICT[EXP]=='stationary')
    data = data[data['ID'].isin(ids)]
    # Read csv with outliers
    outliers = pd.read_csv(OUTLIERS_FILE)
    outliers = outliers[outliers['EXP'] == EXP]

    # Remove outliers
    for i in range(len(outliers)):
        data = data.drop(data[(data['ID'] == outliers['p'].values[i]) & (data['event'] == outliers['trial'].values[i])].index)

    if EXP == 0:
        EXPERIMENTS_DATA['vel0'] = data[(data['dir'] == 0)]
        n_exp += 1
    else:
        EXPERIMENTS_DATA[KEYs[n_exp]] = data[(data['dir'] == -1)]
        n_exp += 1
        EXPERIMENTS_DATA[KEYs[n_exp]] = data[(data['dir'] == 1)]
        n_exp += 1

thresholds_pd = pd.DataFrame(columns=['ID', 'PSE_vel0', 'PSE_pursuitL','PSE_pursuitR','PSE_vel25L','PSE_vel25R','PSE_vel50L','PSE_vel50R', 
                                      'DT25_vel0','DT25_pursuitL','DT25_pursuitR','DT25_vel25L','DT25_vel25R','DT25_vel50L','DT25_vel50R', 
                                      'DT75_vel0','DT75_pursuitL','DT75_pursuitR','DT75_vel25L','DT75_vel25R','DT75_vel50L','DT75_vel50R'])
for id in tqdm.tqdm(ids):
    # Add an empty row to the dataframe with the new participant
    rowIndex = len(thresholds_pd)
    thresholds_pd.loc[rowIndex, 'ID'] = id

    for key in KEYs:
        print('Experiment: ', key)
        # Take only the data for the current participant and current experiment
        all_data = EXPERIMENTS_DATA[key]
        p_data = all_data[all_data['ID'] == id]
        parsed_data = utils.parse_data(p_data, DIR[key])
        result = ps.psignifit(parsed_data, OPTIONS)

        print('fitted parameters: ', result['Fit'][0]   )
        pse = ps.getThreshold(result,0.5)[0]
        dt25 = ps.getThreshold(result,0.25)[0]
        dt75 = ps.getThreshold(result,0.75)[0]

        thresholds_pd.loc[rowIndex, 'PSE_' + key] = pse
        thresholds_pd.loc[rowIndex, 'DT25_' + key] = dt25
        thresholds_pd.loc[rowIndex, 'DT75_' + key] = dt75

# Save the dataframe to a csv file
thresholds_pd.to_csv(OUTPUT_FILE, index=False)
