# Darts Handicapping Respoitory 

This repository contains code that was used to model and evaluate different handicap systems for the game of darts using a Markov Decision Process (MDP) framework.

The majority of the analysis can be found in the `analysis` directory. There are five subdirectories containing major branches of analysis. 

0. `0_analysis_skill` --> Creates the transition probability datasets.
    * `0_0_generate_skill_models.ipynb` --> Prerequisite for all other analysis. Integrates the bivariate Gaussian skill models of all players and saves the corresponding transition probabilities. 
    * `0_1_eda_skill_models.ipynb` --> Tool for visually exploring player Gaussians. Also contains a legend of professional players names and how they are indexed. 
    * `0_2_eda_skill_estimation.ipynb` --> Implements and evaluates our framework for estimating a player's epsilon.  
    * `0_3_eda_skill_estimation.ipynb` --> Implements and evaluates a slightly more sensitive framework for estimating a player's epsilon. 
1. `1_analysis_singleplayer_noturn` --> Solves the simplified MDP (i.e. NOT considering turns) for a single player.
2. `2_analysis_singleplayer_turn` --> Solves the full MDP (i.e. considering turns) for a single player.
3. `3_analysis_twoplayer_ns_turn` --> Uses the optimal policies from the single player MDPs (with turns) and solves a Markov Chain to evaluate the win probabilities for each player at each game state considering both of their scores. There is no optimization here, only policy evaluation. This is why we use the subscript `_ns_` which stands for "non-strategic". 
4. `4_analysis_twoplayer_zsg_turn` --> Optimizes the policies of one or both players in a two-player MDP. For the one player model, uses an expanded MDP formulation with policy iteration. For the two player game, uses a ZSG formulation which iteratively solves MDPs to find the optimal equilibrium policies for both players. 

All data relating to skill models and transition probabilities can be found in the `data_parameter` directory. 
* `ALL_Model_Fits.mat` contains the bivariate Gaussians fit by Haugh & Wang in their 2022 Paper.
* Integrated skill models are also stored in the subdirectory `player_gaussin_fit`.
* `Raw_Data.xlsx` contains the raw throw data that Haugh & Wang used to fit the bivariate Gaussian skill models.

The `result` directory contains solved MDPs (i.e. policies and value functions) as well as datasets generated via simulation. For smaller models, summary .csv folders are saved in addition to .pkl files. 










