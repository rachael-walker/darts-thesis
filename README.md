# Darts Handicapping Respoitory 

This repository contains code that was used to model and evaluate different handicap systems for the game of darts using a Markov Decision Process (MDP) framework.

The majority of the analysis can be found in the `analysis` directory. There are five subdirectories containing major branches of analysis. 

1. `0_analysis_skill` 
2. `1_analysis_singleplayer_noturn`
3. `2_analysis_singleplayer_turn`
4. `3_analysis_twoplayer_ns_turn`
5. `4_analysis_twoplayer_zsg_turn`

All skill model data can be found in the `data_parameter` directory.
* `ALL_Model_Fits.mat` contains the bivariate Gaussians fit by Haugh & Wang in their 2022 Paper.
* Integrated skill models are also stored in the subdirectory `player_gaussin_fit`.
* `Raw_Data.xlsx` contains the raw throw data that Haugh & Wang used to fit the bivariate Gaussian skill models.

The `result` directory contains solved MDPs (i.e. policies and value functions) as well as datasets generated via simulation. For smaller models, summary .csv folders are saved in addition to .pkl files. 










