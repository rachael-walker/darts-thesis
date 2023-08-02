# Darts Handicapping Respoitory 

This repository contains code that was used to model and evaluate different handicap systems for the game of darts using a Markov Decision Process (MDP) framework. 

The key file in this repository is **`helpers.py`** which contains all of the modelling code. This file is called in almost every jupyter notebook in this repository. 

The model code is called in **`generate_turn_data.ipynb`** to produce the csv: `player10_results.csv` which contains the solved optimal policy and value function for every state for a specified group of skill levels. 

The optimal policy and value function are explored in the following exploratory data analysis files: **`eda_policy.ipynb`** and **`eda_value.ipynb`**. 

In addition, there are two class modules: `simulator.py` and `\handicap.py`. 


There are six sub-directories: 
1. **`analysis_noturn`**: contains the exploratory data analysis completed for a simplified "throws" model which does not consider that players may take turns. 
2. **`analysis_skill`**: contains exploratory data analysis around professional player skill models. 
3. **`data_parameter`**: contains the integrated transition probabilities derived from different skill models (both different players and $\epsilon$ values. 
4. **`debugging`**: contains old files that were used in the model development and debugging process. 
5. **`original_code`**: contains the code from Haugh & Wang's 2022 paper "Play Like the Pros" which was instrumental in this repository, which extends (and sometimes directly uses) their original functions. 
6. **`results`**: contains data files that were generated for some of the smaller and simpler models. Many results are not included here because the data was too large to upload to GitHub. 






