# Modeling the Impact of Head-Body Rotations on Audio-Visual Spatial Perception for Virtual Reality Applications

Code and data for *“Modeling the Impact of Head-Body Rotations on Audio-Visual Spatial Perception for Virtual Reality Applications”* ([PDF](https://graphics.unizar.es/papers/TVCG_Modelling_spatial_perception.pdf))

Edurne Bernal-Berdun<sup>1</sup>,	Mateo Vallejo<sup>1</sup>, Qi Sun<sup>2</sup>, Ana Serrano<sup>1</sup>, and Diego Gutierrez<sup>1</sup>

_<sup>1</sup> Universidad de Zaragoza - I3A, <sup>2</sup> New York University_

**IEEE Transactions on Visualization and Computer Graphics (Proc. IEEE VR 2024)**

## Abstract
Humans perceive the world by integrating multimodal sensory feedback, including visual and auditory stimuli, which holds true in virtual reality (VR) environments. Proper synchronization of these stimuli is crucial for perceiving a coherent and immersive VR experience. In this work, we focus on the interplay between audio and vision during localization tasks involving natural head-body rotations. We explore the impact of audio-visual offsets and rotation velocities on users' directional localization acuity for various viewing modes. Using psychometric functions, we model perceptual disparities between visual and auditory cues and determine offset detection thresholds. Our findings reveal that target localization accuracy is affected by perceptual audio-visual disparities during head-body rotations, but remains consistent in the absence of stimuli-head relative motion. We then showcase the effectiveness of our approach in predicting and enhancing users' localization accuracy within realistic VR gaming applications. To provide additional support for our findings, we implement a natural VR game wherein we apply a compensatory audio-visual offset derived from our measured psychometric functions. As a result, we demonstrate a substantial improvement of up to 40% in participants' target localization accuracy. We additionally provide guidelines for content creation to ensure coherent and seamless VR experiences.

Visit our [website](https://graphics.unizar.es/projects/AV_spatial_perception/) for more information and supplementary material.

## Requirements
The code has been tested with:

```
matplotlib==3.3.4
numpy==1.22.4
pandas==1.2.4
psignifit==0.1
seaborn==0.11.1
tqdm==4.59.0
```
## Analysis Scripts
###  Measuring Audio-Visual Spatial Disparity
#### HEAD VELOCITY ANALYSIS
We conducted an analysis of head velocities to ensure adherence to experimental requirements. Outlier velocities, identified using the 1.5×IQR method, constituting 3% of trials, were excluded from the data. The script `velocities.py` performs analysis on head tracking data, generating statistical insights outlined in Table 2 of our paper. It also identifies outlier trials to be removed from the data before psychometric fitting. The outlier points are stored in a CSV file in the desired folder (./data by default).

#### PSYCHOMETRIC FITTING
To visualize psychometric curves derived from pooled participant responses across experimental conditions, use the `fit_psychometric.py` script. Specify the desired condition using the global variable `EXP`, (0 = stationary, 1 = pursuit, 2 = slow reorientation, 3 = fast reorientation). Additionally, individual PSE and DT values can be obtained by executing the `PSE_TH.py` script, which stores values for each condition in a CSV file within the specified folder (default: ./data).

Refer to Section 3 of our paper for further details.

###  Enhancing VR Audio-Visual Target Localization
The script wenstern_grab_gun.py visualizes results from our proof-of-concept application, where PSEs are utilized to improve target localization. It presents a comparison of participant accuracy with and without compensatory offsets (refer to Figure 5 in our paper) and showcases a confusion matrix illustrating participant responses to visual stimuli (refer to Figure 7 in our paper).

Refer to Section 4.1 of our paper for further details.

## Download Data
Data for all participants can be downloaded from the following link: [https://nas-graphics.unizar.es/s/AV_spatial_perception_RAW_data](https://nas-graphics.unizar.es/s/AV_spatial_perception_RAW_data).

Note that scripts can be executed with pre-filtered and processed data, files `filtered_data.zip` and `data_application.zip`. 

## Cite

If you use this work, please consider citing our paper with the following Bibtex code:
```
@article{Bernal-Berdun2024spatial,
        author={Bernal-Berdun, Edurne and Vallejo, Mateo and Sun, Qi and Serrano, Ana and Gutierrez, Diego},
        journal={IEEE Transactions on Visualization and Computer Graphics}, 
        title={Modeling the Impact of Head-Body Rotations on Audio-Visual Spatial Perception for Virtual Reality Applications}, 
        year={2024},
        pages={1-9},
        doi={10.1109/TVCG.2024.3372112}}
```

