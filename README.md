# Darts Handicapping Respoitory 

This repository contains code that was used to model and evaluate different handicap systems for the game of darts using a Markov Decision Process (MDP) framework.

The bulk of our work and exploration can be found in the `analysis` directory. There are five subdirectories containing major branches of analysis. 

0. `0_analysis_skill`: Creates the transition probability datasets.
    - `0_0_generate_skill_models.ipynb`: Prerequisite for all other analysis. Integrates the bivariate Gaussian skill models of all players and saves the corresponding transition probabilities. 
    - `0_1_eda_skill_models.ipynb`: Tool for visually exploring player Gaussians. Also contains a legend of professional players names and how they are indexed. 
    - `0_2_eda_skill_estimation.ipynb`: Implements and evaluates our framework for estimating a player's epsilon.  
    - `0_3_eda_skill_estimation.ipynb`: Implements and evaluates a slightly more sensitive framework for estimating a player's epsilon. 
1. `1_analysis_singleplayer_noturn`: Solves the simplified MDP (i.e. NOT considering turns) for a single player.
    - `1_0_generate_noturn_data.ipynb`: Prerequisite for all other analysis in this directory. Solves the simplified MDP for all desired epsilon-player combinations and saves them in a .csv file (`result/singlegame_results/player10_noturn_results.csv`).
    - `1_1_eda_noturn_value.ipynb`: Analyses the optimal value function of the solved MDPs. 
    - `1_2_eda_noturn_policy.ipynb`: Analyses the optimal policy of the solved MDPs. 
    - `1_3_eda_noturn_handicaps.ipynb`: Evaluates various handicap interventions using solved MDP results. 
    - `1_4_eda_noturn_simulator.ipynb`: Analyses the distributions of simulated game outcomes using different handicap options. Evaluates each intervention more objectively, independent of base MDP framework that was used to generate policies. 
2. `2_analysis_singleplayer_turn`: Solves the full MDP (i.e. considering turns) for a single player.
    - `2_0_generate_turn_data.ipynb`: Prerequisite for all other analysis in this directory. Solves the full MDP for all desired epsilon-player combinations and saves them in a .csv file (`result/singlegame_results/player10_turn_results.csv`).
    - `2_1_eda_turn_value.ipynb`: Analyses the optimal value function of the solved MDPs. 
    - `2_2_eda_turn_policy.ipynb`: Analyses the optimal policy of the solved MDPs. 
    - `2_3_eda_turn_handicaps.ipynb`: Evaluates various handicap interventions using solved MDP results. 
    - `2_4_eda_turn_simulator.ipynb`: Analyses the win probabilities of simulated game outcomes using different handicap options. framework that was used to generate policies. 
    - `2_5_eda_turn_distributions.ipynb`: Analyses the distributions of simulated game outcomes using different handicap options. A drill-down into simulation results of prior file. Evaluates each intervention more objectively, independent of base MDP framework that was used to generate policies. 
3. `3_analysis_twoplayer_ns_turn`: Uses the optimal policies from the single player MDPs (with turns) and solves a Markov Chain to evaluate the win probabilities for each player at each game state considering both of their scores. There is no optimization here, only policy evaluation. This is why we use the subscript `_ns_` which stands for "non-strategic". 
    - `3_0_generate_ns_win_probability_data.ipynb`: Prerequisite for all other analysis in this directory. Evaluates a Markov Chain which models a two-player game and saves the win probability of the weaker player at each state. Results are stored in a .csv file (`result/singlegame_results/player10_turn_results.csv`). *Note that this file will be over 10GB*.
    - `3_1_eda_ns_win_probabilities.ipynb`: Analyses the true win probabilities of the non-strategic policies with different handicap interventions. Allows us to evaluate the true effectiveness of each handicap at balancing competition. Provides a visual map of win probabilities as each player's score changes in relation to their opponent.  
4. `4_analysis_twoplayer_zsg_turn`: Optimizes the policies of one or both players in a two-player MDP. For the one player model, uses an expanded MDP formulation with policy iteration. For the two player game, uses a ZSG formulation which iteratively solves MDPs to find the optimal equilibrium policies for both players. 
    - `4_0_generate_zsg_data.ipynb`: Prerequisite for all other analysis in this directory. Solves the ZSG (and best response) scenario where both (or one) of the players consider their opponent's game state when optimizing their policy. Saves results in large .pkl files (in `result/twoplayer_zsg_results` when both players are being optimized and in `result/twoplayer_br_results` when only one player is being optimized). Note that a single ZSG takes around a day to fully solve and will use substantial RAM. Best response scenarios run much quicker. 
    - `4_1_eda_zsg.ipynb`: Analyses the outcomes of the ZSG and BR scenarios. Shows that the win probabilities and optimal policies are not majorly impacted by both players optimizing. Shows that the decomposed model is a practical option when using a dynamic credit handicap (i.e. the solving each player independently in (2), assuming that they will play their best policy regardless of their opponent, and potentially using (3) to equalize win probabilities using this assumption).

All data relating to skill models and transition probabilities can be found in the `data_parameter` directory. 
* `ALL_Model_Fits.mat` contains the bivariate Gaussians fit by Haugh & Wang in their 2022 Paper.
* `AVG_Model_Fits.mat` contains the bivariate Guassian fits for an aggregate "average" professional player. This was used so as to not bias our analysis too much towards any particular professional player's underlying skill model. 
* Integrated skill models are also stored in the subdirectory `player_gaussin_fit`.
* `Raw_Data.xlsx` contains the raw throw data that Haugh & Wang used to fit the bivariate Gaussian skill models.

The `result` directory is generated once the analysis files are run. This directory contains solved MDPs (i.e. policies and value functions) as well as datasets generated via simulation. For smaller models, summary .csv folders are saved in addition to .pkl files. 

Finally, the main repository contains several helper modules containing the core functions and parameters used for all analysis. 
* `class_handicap.py`: Defines the Handicap class which can be used to quickly and efficiently compute any desired handicap for either the turn or no-turn MDP models (depending on how it's initialized).
* `class_simulator_noturn.py`: Defines a Simulator class which can be used to simulate game outcomes when there is no turn feature. 
* `class_simulator_turn.py`: Defines a Simulator class which can be used to simulate game outcomes when there is a turn feature. 
* `function_helpers_singleplayer.py`: Defines helper functions for all single-player analysis. This includes integrating the underlying skill models to calculate transition probabilities as well as solving the no-turn and turn MDPs (i.e. supports analysis directories 0,1, and 2 directly, which in turn support analysis directories 3 and 4).
* `function_helpers_twoplayer.py`: Defines helper functions for all two-player analysis. This includes two-player policy evaluation using a two-player Markov chain formulation (analysis directory 3) and two-player optimization using a best-response or ZSG formulation (analysis directory 4).
* `function_init_board.py`: Defines board dimensions which are used for integration of skill models. These definitions were taken from Haugh & Wang's original darts code. 
* `function_init_simple_mdp.py`: Defines components of core underlying MDP and provides datastructures which can support interpretation of analysis. Most critically, the action set is defined in this module. 
* `function_plot_board.py`: Defines measurements which are used for all board plotting. 
* `function_tool.py`: Contains helper function to assist with saving and loading .pkl files. These functions were taken from Haugh and Wang's original darts code. 


Would like to thank Martin Haugh & Chun Wang for producing a very clean [codebase](https://github.com/wangchunsem/OptimalDarts) and sharing it publicly. Their original work was instrumental in enabling this handicapping analysis. 




