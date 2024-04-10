import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

def load_and_filter_eyetracking_data(file_path, cvs_files_path=None, file_name='filtered', base=False):

    if os.path.exists(os.path.join(cvs_files_path, file_name + '_data.csv')) and os.path.exists(os.path.join(cvs_files_path, file_name + '_eyetracking_data.csv')):
        return pd.read_csv(os.path.join(cvs_files_path, file_name + '_data.csv')), pd.read_csv(os.path.join(cvs_files_path, file_name + '_eyetracking_data.csv'))

    et_data = pd.DataFrame(columns=['ID','event','dir', 'audio_angle', 'timestamp', 'u_eye', 'v_eye', 'u_head', 'v_head', 'u_cross', 'v_cross'])
    participants_folder_data = os.listdir(file_path)

    data = pd.DataFrame(columns=['ID', 'event', 'dir', 'audio_volume', 'visual_type', 'audio_angle', 'audio_type', 'answer', 'reaction_time', 'valid'])
 
    for p_fd in participants_folder_data:
        # Chek if path string corresponds to a number character
        if not p_fd.isdigit():
            continue

        # Read the cvs file as int
        participant_data = pd.read_csv(os.path.join(file_path, p_fd, p_fd + '.csv'))

        if base:
            mapping = {participant_data.columns[0]: "visual_type", participant_data.columns[1]: "audio_angle", participant_data.columns[2]: "answer", participant_data.columns[3]: "reaction_time"}
            participant_data = participant_data.rename(columns=mapping)
            # Add the participant ID to the data in the first column
            participant_data.insert(0, 'ID', int(p_fd))
            # Insert a new column with the direction of the rotation as 0 
            participant_data.insert(2, 'dir', 0)
            # Insert a new column with the event as the number of the trial
            participant_data.insert(1, 'event', participant_data.index)
            # Insert a new column with the audio volume as 0
            participant_data.insert(3, 'audio_volume', 0)
            # Insert a new column with the audio type as 0
            participant_data.insert(6, 'audio_type', 0)
            # Insert a new column with the valid as True
            participant_data.insert(9, 'valid', True)
        else:
            # Add the participant ID to the data in the first column
            participant_data.insert(0, 'ID', int(p_fd))
            # Change the column names
            participant_data.columns = ['ID', 'event', 'dir', 'audio_volume', 'visual_type', 'audio_angle', 'audio_type', 'answer', 'reaction_time', 'valid']
        
        # remove not valid trials
        participant_data = participant_data[participant_data['valid'] == 1]
        # remove the trials with offset 25 and -25
        participant_data = participant_data[participant_data['audio_angle'] != 25]
        participant_data = participant_data[participant_data['audio_angle'] != -25]

        data = data.append(participant_data, ignore_index=True)

        for et_file in os.listdir(os.path.join(file_path, p_fd, 'eyetracker')):
            
            n_trial = int(et_file.split('_')[1].split('.')[0])

            if not n_trial in participant_data['event']:
                continue
            # Read the cvs file
            et_file_data = pd.read_csv(os.path.join(file_path, p_fd, 'eyetracker', et_file), delimiter=';')
            
            # Filter the data
            # Remove the rows with 'eye_valid_L' != 31 and 'eye_valid_R' != 31 (this flag indicates that the eye data is not valid)
            et_file_data = et_file_data[(et_file_data['eye_valid_L'] == 31) & (et_file_data['eye_valid_R'] == 31)]
            # Remove the rows with 'pupil_diameter_L(mm)' < 0 and 'pupil_diameter_R(mm)' < 0 (this indicates that the sample data is not correct)
            et_file_data['pupil_diameter_L(mm)'] = [float(val.replace(',', '.')) for val in et_file_data['pupil_diameter_L(mm)'].values]
            et_file_data['pupil_diameter_R(mm)'] = [float(val.replace(',', '.')) for val in et_file_data['pupil_diameter_R(mm)'].values]
            et_file_data = et_file_data[(et_file_data['pupil_diameter_L(mm)'] > 0) & (et_file_data['pupil_diameter_R(mm)'] > 0)]
            # Remove the rows with 'openness_L' < 0.85 and 'openness_R' < 0.85 (eye needs to be sufficiently open to get a good sample data)
            o_L = [str(value) for value in et_file_data['openness_L'].values]
            o_R = [str(value) for value in et_file_data['openness_R'].values]
            et_file_data['openness_L'] = [float(val.replace(',', '.')) for val in o_L]
            et_file_data['openness_R'] = [float(val.replace(',', '.')) for val in o_R]
            et_file_data = et_file_data[(et_file_data['openness_L'] > 0.85) & (et_file_data['openness_R'] > 0.85)]
            
            
            # Create a new DataFrame to store the filtered data
            filtered_et_data = pd.DataFrame(columns=['ID','event','dir', 'audio_angle', 'timestamp', 'u_eye', 'v_eye', 'u_head', 'v_head', 'u_cross', 'v_cross'])
            # Add the data to the new DataFrame
            # Set the timestamp of the eyetracking data to start at 0.
        
            filtered_et_data['timestamp'] = et_file_data['time_stamp(ms)']

            if len(filtered_et_data['timestamp'].values) < 1:
                print('Id: ' + p_fd + ' Trial: ' + str(n_trial) + ' No data after invalid data removal')
                continue

            filtered_et_data['timestamp'] = filtered_et_data['timestamp'] - filtered_et_data['timestamp'].values[0]

            filtered_et_data['u_eye'] = np.array([float(value.replace(',','.')) for value in et_file_data['u'].values])
            filtered_et_data['v_eye'] = np.array([float(value.replace(',','.')) for value in et_file_data['v'].values])

            head_forward = np.array([0,0,1,0])

            # Take the elements of worldTolocation matrix
            u_head = []
            v_head = []

            for element in et_file_data['worldToLocalMatrix'].values:
                element = element.replace(',', '.').split(' ')
                matrix = np.zeros((4,4))
                matrix[0,:] = np.array(element[0:4], dtype=np.float32)
                matrix[1,:] = np.array(element[4:8], dtype=np.float32)
                matrix[2,:] = np.array(element[8:12], dtype=np.float32)
                matrix[3,:] = np.array(element[12:16], dtype=np.float32)
                
                head_vector = np.matmul(matrix, head_forward)
                # Normalize the vector
                head_vector = head_vector / np.linalg.norm(head_vector)

                u = (np.arctan2(head_vector[0], head_vector[2]) / (2 * np.pi)) + 0.5
                v = 0.5 - (np.arcsin(head_vector[1]) / np.pi)

                u_head.append(u)
                v_head.append(v)

            filtered_et_data['u_head'] = np.array(u_head)
            filtered_et_data['v_head'] = np.array(v_head)

            filtered_et_data['u_cross'] = np.array([float(value.replace(',','.')) for value in et_file_data['cross_u'].values])
            filtered_et_data['v_cross'] = np.array([float(value.replace(',','.')) for value in et_file_data['cross_v'].values])

            filtered_et_data['ID'] = int(p_fd)
            # assert error if the event is not in the participant_data
            assert n_trial in participant_data['event'].values, 'The event ' + str(n_trial) + ' is not in the participant ' + str(p_fd) + ' data'
            idx = np.where(participant_data['event'] == n_trial)[0][0]
            filtered_et_data['event'] = participant_data['event'].values[idx]
            filtered_et_data['dir'] = participant_data['dir'].values[idx]
            filtered_et_data['audio_angle'] = participant_data['audio_angle'].values[idx]

            et_data = et_data.append(filtered_et_data, ignore_index=True)

    # Save the data in a csv file
    et_data.to_csv(os.path.join(cvs_files_path, file_name + '_eyetracking_data.csv'), index=False)
    data.to_csv(os.path.join(cvs_files_path, file_name +'_data.csv'), index=False)

    return data, et_data

def parse_data(data, dir):
    '''
    Parse data for a given direction (dir) and return a n x 3 matrix of the form:
        [x-value, number correct, number of trials]
        Creates a new row for each participant and offset
    '''
    dat = data[(data['dir'] == dir)]
    offsets = np.unique(dat['audio_angle'].values)

    n_correct = []
    x  =[]
    n_trials = []

    for offset in offsets:
        for p in np.unique(dat['ID'].values):
            answers = dat.loc[(dat['ID'] == p) & (dat['audio_angle'] == offset), 'answer'].values
            answers = np.logical_not(answers)
            n_correct.append(np.sum(answers == False))
            # x.append((offset - np.min(offsets)) / (np.max(offsets) - np.min(offsets)))
            x.append(offset)

            n_trials.append(len(dat.loc[(dat['ID'] == p) & (dat['audio_angle'] == offset), 'answer'].values))
            
    return np.array([x, n_correct, n_trials]).T


def standard_error(data, dir):

    # Compute the standard error for each offset
    dat = data[(data['dir'] == dir)]
    offsets = np.unique(data['audio_angle'].values)
    ids = np.unique(data['ID'].values)
    serror = np.zeros((len(offsets)))

    # Compute the probability of right answer (R=Flase, L=True) for each spatial separation
    for i, offset in enumerate(offsets):
        p_right_participants = np.zeros((len(ids)))
        # Compute the probability of right answer for each participant
        for j,p in enumerate(ids):
            # Get the data for the current participant and spatial separation
            participant_answers = dat.loc[(dat['ID'] == p) & (dat['audio_angle'] == offset), 'answer'].values
            # Set 0 values as 1 and 1 values as 0
            participant_answers = np.logical_not(participant_answers)
            # Compute the probability of right answer
            p_right_participants[j] = np.nanmean(participant_answers == False)
        serror[i] =  np.nanstd(p_right_participants) /np.sqrt(len(p_right_participants))
    return serror


def plotPsych(result,
              dataColor      = [0, 105./255, 170./255],
              plotData       = True,
              lineColor      = [0, 0, 0],
              lineWidth      = 2,
              xLabel         = 'Stimulus Level',
              yLabel         = 'Proportion Correct',
              labelSize      = 15,
              fontSize       = 10,
              fontName       = 'Arial',
              tufteAxis      = False,
              plotAsymptote  = True,
              plotThresh     = True,
              aspectRatio    = False,
              extrapolLength = .2,
              CIthresh       = False,
              dataSize       = 0,
              axisHandle     = None,
              showImediate   = True,
              showse        = False,
              colorse       = 'gray',
              se = None,
              DTarea = False,
              title = None):
    """
    This function produces a plot of the fitted psychometric function with 
    the data.
    """
    
    fit = result['Fit']
    data = result['data']
    options = result['options']
    
    if axisHandle == None: axisHandle = plt.gca()
    try:
        plt.sca(axisHandle)
    except TypeError:
        raise ValueError('Invalid axes handle provided to plot in.')
    
    if np.isnan(fit[3]): fit[3] = fit[2]
    if data.size == 0: return
    if dataSize == 0: dataSize = 10000. / np.sum(data[:,2])
    
    if 'nAFC' in options['expType']:
        ymin = 1. / options['expN']
        ymin = min([ymin, min(data[:,1] / data[:,2])])
    else:
        ymin = 0
    
    
    # PLOT DATA
    #holdState = plt.ishold()
    #if not holdState: 
    #    plt.cla()
    #    plt.hold(True)
    xData = data[:,0]
    if plotData:
        yData = data[:,1] / data[:,2]
        markerSize = np.sqrt(dataSize/2 * data[:,2])
        for i in range(len(xData)):
            plt.plot(xData[i], yData[i], 'o', ms=0.2*markerSize[i], c=dataColor, clip_on=False)
    if showse:
        # plt.fill_between(xData, yData-se, yData+se, color=colorse)
        # Plot se as error bars instead of shaded area
        plt.errorbar(xData, yData, yerr=se, fmt='none', ecolor=colorse)
    
    # PLOT FITTED FUNCTION
    if options['logspace']:
        xMin = np.log(min(xData))
        xMax = np.log(max(xData))
        xLength = xMax - xMin
        x       = np.exp(np.linspace(xMin, xMax, num=1000))
        xLow    = np.exp(np.linspace(xMin - extrapolLength*xLength, xMin, num=100))
        xHigh   = np.exp(np.linspace(xMax, xMax + extrapolLength*xLength, num=100))
        axisHandle.set_xscale('log')
    else:
        xMin = min(xData)
        xMax = max(xData)
        xLength = xMax - xMin
        x       = np.linspace(xMin, xMax, num=1000)
        xLow    = np.linspace(xMin - extrapolLength*xLength, xMin, num=100)
        xHigh   = np.linspace(xMax, xMax + extrapolLength*xLength, num=100)
    
    fitValuesLow  = (1 - fit[2] - fit[3]) * options['sigmoidHandle'](xLow,  fit[0], fit[1]) + fit[3]
    fitValuesHigh = (1 - fit[2] - fit[3]) * options['sigmoidHandle'](xHigh, fit[0], fit[1]) + fit[3]
    fitValues     = (1 - fit[2] - fit[3]) * options['sigmoidHandle'](x,     fit[0], fit[1]) + fit[3]
    
    plt.plot(x,     fitValues,           c=lineColor, lw=lineWidth, clip_on=False)
    plt.plot(xLow,  fitValuesLow,  '--', c=lineColor, lw=lineWidth, clip_on=False)
    plt.plot(xHigh, fitValuesHigh, '--', c=lineColor, lw=lineWidth, clip_on=False)

    # PLOT PARAMETER ILLUSTRATIONS
    # THRESHOLD
    if plotThresh:
        if options['logspace']:
            x = [np.exp(fit[0]), np.exp(fit[0])]
        else:
            x = [fit[0], fit[0]]
        y = [ymin, fit[3] + (1 - fit[2] - fit[3]) * options['threshPC']]
        plt.plot(x, y, '-', c=lineColor)
    # ASYMPTOTES
    if plotAsymptote:
        plt.plot([min(xLow), max(xHigh)], [1-fit[2], 1-fit[2]], ':', c=lineColor, clip_on=False)
        plt.plot([min(xLow), max(xHigh)], [fit[3], fit[3]],     ':', c=lineColor, clip_on=False)
    # CI-THRESHOLD
    if CIthresh:
        CIs = result['conf_Intervals']
        y = np.array([fit[3] + .5*(1 - fit[2] - fit[3]) for i in range(2)])
        plt.plot(CIs[0,:,0],               y,               c=lineColor)
        plt.plot([CIs[0,0,0], CIs[0,0,0]], y + [-.01, .01], c=lineColor)
        plt.plot([CIs[0,1,0], CIs[0,1,0]], y + [-.01, .01], c=lineColor)

    if DTarea:
        plt.fill_between([min(xLow),max(xHigh)], 0.25, 0.75, color='lavenderblush', alpha=0.7, zorder=0)
        plt.axhline(y=0.5, color='darkslategray', linestyle='--', linewidth=1, zorder=0)
        plt.axhline(y=0.75, color='darkslategray', linestyle='--', linewidth=1, zorder=0)
        plt.axhline(y=0.25, color='darkslategray', linestyle='--', linewidth=1, zorder=0)
    
    #AXIS SETTINGS
    plt.axis('tight')
    plt.tick_params(labelsize=fontSize)
    plt.xlim(min(xLow), max(xHigh))
    plt.xlabel(xLabel, fontname=fontName, fontsize=labelSize)
    plt.ylabel(yLabel, fontname=fontName, fontsize=labelSize)
    if aspectRatio: axisHandle.set_aspect(2/(1 + np.sqrt(5)))

    plt.ylim([ymin, 1])
    # tried to mimic box('off') in matlab, as box('off') in python works differently
    plt.tick_params(direction='out',right=False,top=False)
    for side in ['top','right']: axisHandle.spines[side].set_visible(False)
    plt.gca().xaxis.set_major_formatter(ScalarFormatter())
    plt.ticklabel_format(style='sci',scilimits=(-2,4))
    
    #plt.hold(holdState)
    if (showImediate):
        plt.show(block=False)
    return axisHandle
