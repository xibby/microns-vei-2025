# neuro-static
Static networks for neural population prediction

## Short model explanation
Model has 4 components:
* core: Receives image, passes it through some convolutional layers. Input is H x W, output is H x W x C
* shifter: Receives eye position and computes expected shift for all cells. Input is a tuple, output is a tuple.
* readout: Per cell, it applies a weighted mask to the output of core and outputs a single value. The mask is shifted using the output of shifter. Input is HxWxC and a tuple, output is NCELLS.
* modulator: Receives behavioral data, computes a gain/modulator factor per cell and rescales/multiplies the output of readout with it. Input is a triple (pupil_dilation, dpupil/dt and treadmill) and NCELLS, output is NCELLS.
### During evaluation:
We have to decide what values for the eye position, and behavioral variables will be sent to the shifter and modulator, respectively.
#### eye_pos
Controls the overall shift applied to every single cell (displacement of a cell RF in the input image)
* None: Assumes the mouse was looking straight in the middle of the monitor.
* training_mean_eye_position (0, 0 if data was Normalized): shifts all cells to the average pupil position during training
#### behavior
* None: Output of readout is sent as is.
* training_mean_behavior: scales each cell's output to their scale in the training set.

## Run-through of MEI tables
### Prepare data for training
Final outpus is data_schemas.StaticMultiDataset ==> dataset usable to train stuff
`from neuro_data.static_images import data_schemas`
1. data_schemas.ExcludedTrial ==> List of malformed trials
2. data_schemas.StaticScan ==> List of static scans
3. data_schemas.ConditionTier ==> Split each image into train/val/test split
4. data_schemas.Frame ==> all (resized) images (in ConditionTier)
5. data_schemas.InputResponse ==> Response of each cell to each image (ResponseBlock is num_trials x num_cells)
6. data_schemas.Eye() ==> average pupil dilation for each frame
7. data_schemas.Treadmill() ==> treadmill velocity during each frame (averaged over the 0.5 secs)
8. data_schemas.StaticMultiDataset.fill() ==> dataset usable to train stuff # needs to add the key in `selection` variable inside fill
9. anatomy.AreaMembership ==> What area each cell belongs to (Manolis)
10. anatomy.LayerMembership ==> What layer each cell belongs to (Manolis)
11. (if needed) neuro_data.static_images.configs.DataConfig.fill() ==> All possible data combinations (whether data will be normalized or not, whether stimulus was Frame or not, whether they come from L2/3 or L4, whether they come from V1 or LM, among others). See neuro_data.configs.ConfigData.CorrectedAreaLayer.content
    

### Train model(s)
```python
from staticnet_experiments import models, configs

# Select the architecture and training desired
candidates = [configs.CoreConfig.StackedLinearGaussianLaplace & {'gauss_bias': 0, 'gauss_sigma': 0.5, 'input_kern': 15, 'hidden_kern': 7}, 
              configs.CoreConfig.GaussianLaplace & {'gauss_bias': 0, 'gauss_sigma': 0.5, 'input_kern': 15, 'hidden_kern': 7}]
model_candidates = configs.NetworkConfig.CorePlusReadout & candidates & (configs.TrainConfig.Default & {'batch_size': 60}) & (configs.DataConfig.CorrectedAreaLayer() & {'stimulus_type': 'stimulus.Frame', 'exclude': '', 'layer': 'L2/3', 'brain_area': 'all', 'normalize_per_image': False}) # 8 core configs x 2 readout configs x 4 seeds = 64 models to train

# Train it
models.Model.populate({'group_id': 23}, model_candidates, reserve_jobs=True) # ==> Trained models
```

### Generate MEIs
`from staticnet import multi_mei`
1. multi_mei.TargetModel ==> Models for which to generate MEIs (just the PK of models.Model)
2. multi_mei.TargetDataset ==> List of datasets to process as well as area and layer of each unit.
3. stats.Oracle ==> Oracle correlation for repeated images.
4. multi_mei.OracleRankedUnit ==> Ranks all units from TargetDataset based on their oracle.
5. multi_mei.ModelGroup ==> Save best linear and cnn model (to be used for unit selection)
6. multi_mei.CorrectedHighUnitSelection ==> Select best units. MEI generation restricted to these.
7. multi_mei.MEI ==> Generate the MEIs
8. [Optionally] multi_mei.TightMEIMask ==> Mask around the MEI for plotting and to compute diversity
      
### Computing confusion matrix (main result of the paper, Fig 4)
`from staticnet_analyses import closed_loop`
1. closed_loop.ClosedLoopScan =-> List of scans to analyze
2. closed_loop.ProximityCellMatch ==> Match cells based on distance from StackCoordinates
3. closed_loop.BestProximityCellMatch ==> Find best cell in scan corresponding to each MEI unit (using majority vote over all stacks)
4. closed_loop.NormalizedConfusion ==> Response of each cell (normalized by dividing to std_per_cell) to its own MEI and that of all others
5. closed_loop.SummaryConfusion ==> Confusion matrix across all scans (using NormalizedConfusion)
6. closed_loop.SummaryConfusionDiagonalEntry ==> Response of each unit to its own MEI and some statistics that will be used for hypothesis testing.
7. closed_loop.ZScoreConfusion, ZSummaryConfusion, ZSummaryConfusionDiagonalEntry = > Confusion matrix using z-scores responses [(response - mean_per_cell)/ std_per_cell]

#### Generating new stimuli (for the closed_loop)
##### For monitor calibration
1. multi_mei.ProcessedMonitorCalibration ==> pixel intensities to luminance fit
2. multi_mei.ClosestCalibration ==> closest monitor calibration for our scans of interest

##### To generate new stimulus
1. stimulus.TargetStaticGroupId == > What group id is gonna be used for the stimulus
2. closed_loop.fill_multimei_stimulus ==> Generate MEI vs RF stimulus and write it down to the stimulus dd
3. closed_loop.fill_gabor_stimulus ==> Generate MEI vs Gabor stimulus and write it down to the stimulus db
4. closed_loop.fill_tight_mei_vs_imagenet_stimulus ==> Fil masked MEI vs masked ImageNet
