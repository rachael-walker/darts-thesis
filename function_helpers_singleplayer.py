#%%
import os
import time

import numpy as np
import scipy.io as sio
from scipy.stats import multivariate_normal

import function_init_board as fb
import function_tool as ft
import function_plot_board as ib
import function_init_simple_mdp as imdp

import numpy as np
np.set_printoptions(precision=4)
np.set_printoptions(linewidth=300)
np.set_printoptions(threshold=300)

import torch
torch.set_printoptions(precision=4)
torch.set_printoptions(linewidth=300)
torch.set_printoptions(threshold=300)

R = fb.R ## radius of the dartboard 170
grid_num = fb.grid_num ## 341


#%%
def evaluate_score_probability(playerID_list,epsilon=1,f_density_grid_pixel_per_mm=0.1):    
    """
    Players' fitted skill models are contained in ALL_Model_Fits.mat.
    A dart throw landing follows a bivariate Gaussian distribution with the mean(center) as the aiming location and the covariance matrix given in the fitted model. 
    This function conducts a numerical integration to evaluate the hitting probability of each score (each score segment in the dartboard) associated with each aiming location on the 1mm-grid.
    
    Args: 
        A list of playerID to evaluate, e.g., [1,2] for player1 (Anderson) and player2 (Aspinall).
    
    Returns: 
        A dict of four numpy arrays: prob_grid_singlescore, prob_grid_doublescore, prob_grid_triplescore, prob_grid_bullscore.
        prob_grid_singlescore[xi,yi,si] is of size 341*341*20.  (prob_grid_doublescore and prob_grid_triplescore have the similar structure.)
        xi and yi are the x-axis and y-axis indexes (starting from 0) of the square 1mm grid enclosing the circle dartboard.
        si = 0,1,...,19 for score S1,S2,...,S20
        prob_grid_bullscore[xi,yi,si] is of size 341*341*2, where si=0 represents SB and si=1 represents DB 
        For example, when aiming at the center of the dartboard,
        prob_grid_singlescore[xi=170,yi=170,si=9] is the probability of hitting S10 
        prob_grid_doublescore[xi=170,yi=170,si=0] is the probability of hitting D1 
        prob_grid_triplescore[xi=170,yi=170,si=7] is the probability of hitting T8 
        prob_grid_bullscore[xi=170,yi=170,si=0] is the probability of hitting SB
        prob_grid_bullscore[xi=170,yi=170,si=1] is the probability of hitting DB
        
        Results are stored in the folder ./data_parameter/player_gaussin_fit/grid_full
                
    """

    result_dir = fb.data_parameter_dir + '/grid_full'
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)
    player_parameter = sio.loadmat('./data_parameter/ALL_Model_Fits.mat')
    
    ## 1mm-width grid of 341*341 aiming locations (a sqaure enclosing the circle dart board)
    [xindex, yindex, xgrid, ygrid, grid_num] = fb.get_1mm_grid()
    
    ## 0.5mm-width grid of 681*681 locations for evaluating the PDF of the fitted Gaussian distribution
    # f_density_grid_pixel_per_mm = 2
    # f_density_grid_pixel_per_mm = 0.1
    # f_density_grid_pixel_per_mm = 0.5
    #f_density_grid_pixel_per_mm = 1
    f_density_grid_num = int(2*fb.R*f_density_grid_pixel_per_mm) + 1
    f_density_grid_width = 1.0/f_density_grid_pixel_per_mm
    f_density_constant = f_density_grid_width*f_density_grid_width
    print('f_density_grid_num={} f_density_grid_width={}'.format(f_density_grid_num, f_density_grid_width))
    
    ## f_density_grid x coordinate left to right increasing
    f_density_xindex = range(f_density_grid_num)
    f_density_xgrid =  np.arange(f_density_grid_num) * f_density_grid_width - fb.R
    ## y coordinate top to bottom increasing
    f_density_yindex = f_density_xindex[:]
    f_density_ygrid = f_density_xgrid[:]
    
    # build f_density_grid, x is the horizon axis (column index) and y is the vertical axis (row index). Hence, y is at first
    y, x = np.mgrid[-fb.R:fb.R+0.1*f_density_grid_width:f_density_grid_width, -fb.R:fb.R+0.1*f_density_grid_width:f_density_grid_width]
    pos = np.dstack((x, y))
    
    ## score information on the f_density_grid
    singlescore_grid = np.zeros((f_density_grid_num, f_density_grid_num), dtype=np.int8)
    doublescore_grid = np.zeros((f_density_grid_num, f_density_grid_num), dtype=np.int8)
    triplescore_grid = np.zeros((f_density_grid_num, f_density_grid_num), dtype=np.int8)
    bullscore_grid = np.zeros((f_density_grid_num, f_density_grid_num), dtype=np.int8)
    for xi in f_density_xindex:
        for yi in f_density_yindex:
            singlescore_grid[yi,xi] = fb.get_score_singleonly(f_density_xgrid[xi], f_density_ygrid[yi])
            doublescore_grid[yi,xi] = fb.get_score_doubleonly(f_density_xgrid[xi], f_density_ygrid[yi])
            triplescore_grid[yi,xi] = fb.get_score_tripleonly(f_density_xgrid[xi], f_density_ygrid[yi])
            bullscore_grid[yi,xi] = fb.get_score_bullonly(f_density_xgrid[xi], f_density_ygrid[yi])
    singlescore_coordinate_dic = {}
    doublescore_coordinate_dic = {}
    triplescore_coordinate_dic = {}
    bullscore_coordinate_dic = {}
    
    ## coordinate for each score
    for si in range(20):
        singlescore_coordinate_dic[si] = np.where(singlescore_grid==fb.singlescorelist[si])
        doublescore_coordinate_dic[si] = np.where(doublescore_grid==fb.doublescorelist[si])
        triplescore_coordinate_dic[si] = np.where(triplescore_grid==fb.triplescorelist[si])
    bullscore_coordinate_dic[0] = np.where(bullscore_grid==fb.bullscorelist[0])
    bullscore_coordinate_dic[1] = np.where(bullscore_grid==fb.bullscorelist[1])
    
    ## 
    for playerID in playerID_list:

        player_index = playerID - 1

        name_pa = 'player{}'.format(playerID)
        result_filename = result_dir + '/' + 'e{}_'.format(epsilon) + '{}_gaussin_prob_grid.pkl'.format(name_pa)
        print('\ncomputing {}'.format(result_filename))
        
        ## new result grid    
        prob_grid_singlescore = np.zeros((grid_num, grid_num, fb.singlescorelist_len))
        prob_grid_doublescore = np.zeros((grid_num, grid_num, fb.doublescorelist_len))
        prob_grid_triplescore = np.zeros((grid_num, grid_num, fb.triplescorelist_len))
        prob_grid_bullscore = np.zeros((grid_num, grid_num, fb.bullscorelist_len))
        
        #### conduct a numerical integration to evaluate the hitting probability for each score associated with the given aiming location
        time1 = time.time()
        for xi in xindex:
            ##print(xi)
            for yi in yindex:
                ## select the proper Gaussian distribution according to the area to which the aiming location belongs
                mu = [xgrid[xi], ygrid[yi]]
                score, multiplier = fb.get_score_and_multiplier(mu)
                if (score==60 and multiplier==3): ##triple 20
                    covariance_matrix = player_parameter['ModelFit_T20'][0, player_index][2] * epsilon
                elif (score==57 and multiplier==3): ##triple 19
                    covariance_matrix = player_parameter['ModelFit_T19'][0, player_index][2] * epsilon
                elif (score==54 and multiplier==3): ##triple 18
                    covariance_matrix = player_parameter['ModelFit_T18'][0, player_index][2] * epsilon
                elif (score==51 and multiplier==3): ##triple 17
                    covariance_matrix = player_parameter['ModelFit_T17'][0, player_index][2] * epsilon
                elif (score==50 and multiplier==2): ##double bull
                    covariance_matrix = player_parameter['ModelFit_B50'][0, player_index][2] * epsilon
                else:
                    covariance_matrix = player_parameter['ModelFit_All_Doubles'][0, player_index][2] * epsilon
                        
                ## f_density_grid is the PDF of the fitted Gaussian distribution
                rv = multivariate_normal(mu, covariance_matrix)
                f_density_grid = rv.pdf(pos)
            
                ## check score and integrate density
                for si in range(20):
                    prob_grid_singlescore[xi,yi,si] = f_density_grid[singlescore_coordinate_dic[si]].sum()*f_density_constant
                    prob_grid_doublescore[xi,yi,si] = f_density_grid[doublescore_coordinate_dic[si]].sum()*f_density_constant
                    prob_grid_triplescore[xi,yi,si] = f_density_grid[triplescore_coordinate_dic[si]].sum()*f_density_constant
                prob_grid_bullscore[xi,yi,0] = f_density_grid[bullscore_coordinate_dic[0]].sum()*f_density_constant
                prob_grid_bullscore[xi,yi,1] = f_density_grid[bullscore_coordinate_dic[1]].sum()*f_density_constant
                
        result_dic = {'prob_grid_singlescore':prob_grid_singlescore, 'prob_grid_doublescore':prob_grid_doublescore,'prob_grid_triplescore':prob_grid_triplescore, 'prob_grid_bullscore':prob_grid_bullscore}    
        time2 = time.time()
        print('computation is done in {} seconds'.format(time2-time1))    
        ft.dump_pickle(result_filename, result_dic, printflag=True)
    
    return result_dic


#%%
## 2-dimension probability grid
def load_aiming_grid(playername_filename, epsilon=1, data_parameter_dir=fb.data_parameter_dir, grid_version=fb.grid_version, count_bull=True):
    """
    Load 2-dimensional numpy arrays of hitting probability from files
    Each row of aiming_grid is the (x-index, y-index) of an aiming location. 
    For each aiming location, the corresponding row in prob_grid_singlescore (same row index as that in aiming_grid) contains the hitting probability of score S1,...,S20.    
    (prob_grid_doublescore for D1,...,D20, prob_grid_triplescore for T1,...,T20,, prob_grid_bullscore for SB,DB)
    prob_grid_normalscore has 61 columns and contains the aggregated hitting probability of score 0,1...,60. For example, P(score 18) = P(S18) + P(D9) + P(T6)

    """

    if playername_filename.startswith('player'):
        filename = data_parameter_dir+'/grid_{}/{}_e{}_gaussin_prob_grid_{}.pkl'.format(grid_version, playername_filename, epsilon, grid_version)
    elif playername_filename.startswith('t'):
        filename = data_parameter_dir+'/grid_{}/t_gaussin_prob_grid_{}.pkl'.format(grid_version, grid_version)
    else:    
        filename = playername_filename    

    result_dic = ft.load_pickle(filename, printflag=True)
    aiming_grid = result_dic['aiming_grid']
    prob_grid_normalscore = result_dic['prob_grid_normalscore'] 
    prob_grid_singlescore = result_dic['prob_grid_singlescore']
    prob_grid_doublescore = result_dic['prob_grid_doublescore']
    prob_grid_triplescore = result_dic['prob_grid_triplescore']
    prob_grid_bullscore = result_dic['prob_grid_bullscore']
    
    ## default setting counts bull score
    if count_bull:
        prob_grid_normalscore[:,fb.score_SB] += prob_grid_bullscore[:,0]
        prob_grid_normalscore[:,fb.score_DB] += prob_grid_bullscore[:,1]
    else:
        print('bull score in NOT counted in prob_grid_normalscore')
        
    return [aiming_grid, prob_grid_normalscore, prob_grid_singlescore, prob_grid_doublescore, prob_grid_triplescore, prob_grid_bullscore]

#%%
## 3-dimension probability grid
def load_prob_grid(playername_filename, epsilon=1, data_parameter_dir=fb.data_parameter_dir, grid_version='full'):
    """
    Load 3-dimensional numpy arrays of size 341*341*si (the 340mmX340mm square grid enclosing the dartboard).
    Generate prob_grid_normalscore which has 61 columns and contains the aggregated hitting probability of score 0,1...,60. For example, P(score 18) = P(S18) + P(D9) + P(T6)
    """
    
    if playername_filename.startswith('player'):
        filename = data_parameter_dir+'/grid_{}/e{}_{}_gaussin_prob_grid.pkl'.format(grid_version, epsilon, playername_filename)
    else:    
        filename = playername_filename        
    
    # Added in min / max to set to reasonable probability values
    prob_grid_dict = ft.load_pickle(filename, printflag=True)    
    prob_grid_singlescore = prob_grid_dict['prob_grid_singlescore']
    prob_grid_doublescore = prob_grid_dict['prob_grid_doublescore']
    prob_grid_triplescore = prob_grid_dict['prob_grid_triplescore']
    prob_grid_bullscore = prob_grid_dict['prob_grid_bullscore']
    
    ## prob_grid_singlescore has 61 columns and contains the aggregated hitting probability of score 0,1...,60. For example, P(score 18) = P(S18) + P(D9) + P(T6)
    ## Bull score is NOT included yet !!
    prob_grid_normalscore = np.zeros((grid_num, grid_num, 61))
    for temp_s in range(1,61):
        if temp_s <= 20:
            prob_grid_normalscore[:,:,temp_s] = prob_grid_singlescore[:,:,temp_s-1]
        if temp_s%2 == 0 and temp_s <= 40:
            prob_grid_normalscore[:,:,temp_s] = prob_grid_normalscore[:,:,temp_s] + prob_grid_doublescore[:,:,temp_s//2-1]
        if temp_s%3 == 0:
            prob_grid_normalscore[:,:,temp_s] = prob_grid_normalscore[:,:,temp_s] + prob_grid_triplescore[:,:,temp_s//3-1]
    ## prob of hitting zero
    prob_grid_normalscore[:,:,0] =  np.maximum(0, 1-prob_grid_normalscore[:,:,1:].sum(axis=2)-prob_grid_bullscore.sum(axis=2))
    
    return [prob_grid_normalscore, prob_grid_singlescore, prob_grid_doublescore, prob_grid_triplescore, prob_grid_bullscore]


#%%
def get_aiming_grid_custom_tokens():

    temp_num = len(imdp.a_throw_list) + len(imdp.a_token_list)

    aiming_grid = np.zeros((temp_num,2), dtype=np.int32)
        
    ## x,y location in [-170,170]. x,y index in [0,340].
    ## center of DB is the first one 
    temp_index = 0
    
    for a in imdp.a_throw_list: 
      temp_x = imdp.actions[a]['gaussian_centroid'][0]
      temp_y = imdp.actions[a]['gaussian_centroid'][1]
      aiming_grid[temp_index] = [temp_x+170, temp_y+170]
      temp_index += 1

    ## post processing
    aiming_grid_throws = temp_index #+ 1

    for a in imdp.a_token_list:
        temp_x = imdp.token_actions[a]['gaussian_centroid'][0]
        temp_y = imdp.token_actions[a]['gaussian_centroid'][1]
        aiming_grid[temp_index] = [temp_x+170, temp_y+170]
        temp_index += 1

    ## post processing
    aiming_grid_num = temp_index #+ 1
    aiming_grid = aiming_grid[:aiming_grid_num,:]
    aiming_grid = np.maximum(aiming_grid, 0)
    aiming_grid = np.minimum(aiming_grid, grid_num-1)

    ## return probability
    prob_grid_normalscore_new = np.zeros((aiming_grid_num, 61))
    prob_grid_singlescore_new = np.zeros((aiming_grid_num, 20))
    prob_grid_doublescore_new = np.zeros((aiming_grid_num, 20))
    prob_grid_triplescore_new = np.zeros((aiming_grid_num, 20))
    prob_grid_bullscore_new = np.zeros((aiming_grid_num, 2))    
    
    # regular actions 
    for temp_index in range(aiming_grid_throws):
        prob_grid_normalscore_new[temp_index,0] = 1

    for temp_index in range(aiming_grid_throws,aiming_grid_num):

        a_index = temp_index - aiming_grid_throws

        a = imdp.a_token_list[a_index]
        rv = imdp.result_values[a]

        is_single = 'S' in a
        is_double = 'D' in a
        is_triple = 'T' in a 
        is_bull = 'B' in a 

        prob_grid_normalscore_new[temp_index,rv] = 1

        if is_bull: 
            prob_grid_normalscore_new[temp_index,rv] = 0
            if 'D' in a: 
                prob_grid_bullscore_new[temp_index,1] = 1
            else: 
                prob_grid_bullscore_new[temp_index,0] = 1

        elif is_single: 
            prob_grid_singlescore_new[temp_index,rv-1] = 1
        
        elif is_double:
            prob_grid_doublescore_new[temp_index,int((rv/2)-1)] = 1
        
        else:
            prob_grid_triplescore_new[temp_index,int((rv/3)-1)] = 1
    
    return [aiming_grid, prob_grid_normalscore_new, prob_grid_singlescore_new, prob_grid_doublescore_new, prob_grid_triplescore_new, prob_grid_bullscore_new]

#%%
## save custom action set with tokens
def save_aiming_grid_custom_tokens(): 

    grid_version_result = 'custom_tokens'

    print('generate and save action set grid_version={}'.format(grid_version_result))
  
    [aiming_grid, prob_grid_normalscore, prob_grid_singlescore, prob_grid_doublescore, prob_grid_triplescore, prob_grid_bullscore] = get_aiming_grid_custom_tokens()
    
    postfix = ''
    info = 'SB={} DB={} R1={} postfix={} skillmodel=full grid_version={}'.format(fb.score_SB, fb.score_DB, fb.R1, postfix, grid_version_result)
    
    result_dic = {}
    result_dic['info'] = info
    result_dic['aiming_grid'] = aiming_grid
    result_dic['prob_grid_normalscore'] = prob_grid_normalscore
    result_dic['prob_grid_singlescore'] = prob_grid_singlescore
    result_dic['prob_grid_doublescore'] = prob_grid_doublescore
    result_dic['prob_grid_triplescore'] = prob_grid_triplescore
    result_dic['prob_grid_bullscore'] = prob_grid_bullscore
    
    result_dir = fb.data_parameter_dir + '/grid_{}'.format(grid_version_result)
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)        
    result_filename = result_dir + '/t_gaussin_prob_grid_{}.pkl'.format(grid_version_result)
    ft.dump_pickle(result_filename, result_dic, printflag=True)
    
    print()
    return

#%%
def get_aiming_grid_custom_no_tokens(playername_filename, epsilon=1, data_parameter_dir=fb.data_parameter_dir, grid_version='full'):

    temp_num = len(imdp.a_throw_list) + len(imdp.a_token_list)
    
    [prob_grid_normalscore, prob_grid_singlescore, prob_grid_doublescore, prob_grid_triplescore, prob_grid_bullscore] = load_prob_grid(playername_filename, epsilon=epsilon, data_parameter_dir=data_parameter_dir, grid_version=grid_version)    

    # temp_num = 1000
    aiming_grid = np.zeros((temp_num,2), dtype=np.int32)
        
    ## x,y location in [-170,170]. x,y index in [0,340].
    ## center of DB is the first one 
    temp_index = 0
    
    for a in imdp.a_throw_list: 
      temp_x = imdp.actions[a]['gaussian_centroid'][0]
      temp_y = imdp.actions[a]['gaussian_centroid'][1]
      aiming_grid[temp_index] = [temp_x+170, temp_y+170]
      temp_index += 1

    ## post processing
    aiming_grid_throws = temp_index #+ 1

    for a in imdp.a_token_list:
        temp_x = imdp.token_actions[a]['gaussian_centroid'][0]
        temp_y = imdp.token_actions[a]['gaussian_centroid'][1]
        aiming_grid[temp_index] = [temp_x+170, temp_y+170]
        temp_index += 1

    ## post processing
    aiming_grid_num = temp_index #+ 1
    aiming_grid = aiming_grid[:aiming_grid_num,:]
    aiming_grid = np.maximum(aiming_grid, 0)
    aiming_grid = np.minimum(aiming_grid, grid_num-1)

    ## return probability
    prob_grid_normalscore_new = np.zeros((aiming_grid_num, 61))
    prob_grid_singlescore_new = np.zeros((aiming_grid_num, 20))
    prob_grid_doublescore_new = np.zeros((aiming_grid_num, 20))
    prob_grid_triplescore_new = np.zeros((aiming_grid_num, 20))
    prob_grid_bullscore_new = np.zeros((aiming_grid_num, 2))    
    
    # regular actions 
    for temp_index in range(aiming_grid_throws):
        prob_grid_normalscore_new[temp_index,:] = prob_grid_normalscore[aiming_grid[temp_index,0], aiming_grid[temp_index,1], :]
        prob_grid_singlescore_new[temp_index,:] = prob_grid_singlescore[aiming_grid[temp_index,0], aiming_grid[temp_index,1], :]
        prob_grid_doublescore_new[temp_index,:] = prob_grid_doublescore[aiming_grid[temp_index,0], aiming_grid[temp_index,1], :]
        prob_grid_triplescore_new[temp_index,:] = prob_grid_triplescore[aiming_grid[temp_index,0], aiming_grid[temp_index,1], :]
        prob_grid_bullscore_new[temp_index,:] = prob_grid_bullscore[aiming_grid[temp_index,0], aiming_grid[temp_index,1], :]
    
    # token actions
    for temp_index in range(aiming_grid_throws,aiming_grid_num):
        prob_grid_normalscore_new[temp_index,0] = 1

    return [aiming_grid, prob_grid_normalscore_new, prob_grid_singlescore_new, prob_grid_doublescore_new, prob_grid_triplescore_new, prob_grid_bullscore_new]

#%%
## save custom action set with no tokens
def save_aiming_grid_custom_no_tokens(playerID_list, epsilon=1,grid_version='full'):    
    grid_version_result = 'custom_no_tokens'
    print('generate and save action set grid_version={}'.format(grid_version_result))
    for playerID in playerID_list:
        name_pa = 'player{}'.format(playerID)    
        [aiming_grid, prob_grid_normalscore, prob_grid_singlescore, prob_grid_doublescore, prob_grid_triplescore, prob_grid_bullscore] = get_aiming_grid_custom_no_tokens(name_pa, epsilon=epsilon, data_parameter_dir=fb.data_parameter_dir, grid_version=grid_version)
        
        postfix = ''
        info = 'SB={} DB={} R1={} postfix={} skillmodel=full grid_version={}'.format(fb.score_SB, fb.score_DB, fb.R1, postfix, grid_version_result)
        
        result_dic = {}
        result_dic['info'] = info
        result_dic['aiming_grid'] = aiming_grid
        result_dic['prob_grid_normalscore'] = prob_grid_normalscore
        result_dic['prob_grid_singlescore'] = prob_grid_singlescore
        result_dic['prob_grid_doublescore'] = prob_grid_doublescore
        result_dic['prob_grid_triplescore'] = prob_grid_triplescore
        result_dic['prob_grid_bullscore'] = prob_grid_bullscore
        
        result_dir = fb.data_parameter_dir + '/grid_{}'.format(grid_version_result)
        if not os.path.isdir(result_dir):
            os.makedirs(result_dir)        
        result_filename = result_dir + '/{}_e{}_gaussin_prob_grid_{}.pkl'.format(name_pa, epsilon, grid_version_result)
        ft.dump_pickle(result_filename, result_dic, printflag=True)
    
    print()
    return


def init_data_structures():
    optimal_value_rt3 = np.zeros(502) #vector: optimal value for the beginning state of each turn (rt=3)
    optimal_value_dic = {} ## first key: score=0,2,...,501, second key: remaining throws=3,2,1
    optimal_action_index_dic = {}
    num_iteration_record = np.zeros(502, dtype=np.int32)
    
    state_len_vector = np.zeros(4, dtype=np.int32)
    state_value  = [None]  ## optimal value (expected # of turns to finish the game) for each state in the current playing turn
    state_action = [None]  ## aimming locations for for each state in the current playing turn
    action_diff  = [None]
    value_relerror = np.zeros(4)
    for rt in [1,2,3]:
        ## for rt=3: possible score_gained = 0
        ## for rt=2: possible score_gained = 0,1,...,60
        ## for rt=1: possible score_gained = 0,1,...,120
        this_throw_state_len = fb.maxhitscore*(3-rt) + 1
        state_value.append(np.ones(this_throw_state_len)*fb.largenumber)
        state_action.append(np.ones(this_throw_state_len, np.int32)*fb.infeasible_marker)
        action_diff.append(np.ones(this_throw_state_len))
    state_value_update = ft.copy_numberarray_container(state_value)
    state_action_update = ft.copy_numberarray_container(state_action)

    return optimal_value_rt3, optimal_value_dic, optimal_action_index_dic, num_iteration_record, state_len_vector, state_value, state_action, action_diff, value_relerror, state_value_update, state_action_update

def init_probabilities( aiming_grid, prob_grid_normalscore_nt, prob_grid_doublescore_nt, prob_grid_bullscore_nt, prob_grid_normalscore_t, prob_grid_doublescore_t, prob_grid_bullscore_t):
    
    # aiming grid
    num_aiming_location = aiming_grid.shape[0]
    prob_normalscore_nt = prob_grid_normalscore_nt
    prob_normalscore_t = prob_grid_normalscore_t
    prob_doublescore_dic_nt = {}
    prob_doublescore_dic_t = {}
    for doublescore_index in range(20):
        doublescore = 2*(doublescore_index+1)
        prob_doublescore_dic_nt[doublescore] = np.array(prob_grid_doublescore_nt[:,doublescore_index])
        prob_doublescore_dic_t[doublescore] = np.array(prob_grid_doublescore_t[:,doublescore_index])
    prob_DB_nt = np.array(prob_grid_bullscore_nt[:,1])
    prob_DB_t = np.array(prob_grid_bullscore_t[:,1])

    ## the probability of not bust for each action given score_max=i (score_remain=i+2)
    prob_bust_dic_nt = {}
    prob_notbust_dic_nt = {}
    prob_bust_dic_t = {}
    prob_notbust_dic_t = {}
    for score_max in range(60):    
        ## transit to next throw or turn
        prob_notbust_nt = prob_grid_normalscore_nt[:,0:score_max+1].sum(axis=1)
        prob_notbust_t = prob_grid_normalscore_t[:,0:score_max+1].sum(axis=1)
        ## transit to the end of game
        score_remain = score_max + 2
        if (score_remain == fb.score_DB):
            prob_notbust_nt += prob_DB_nt
            prob_notbust_t += prob_DB_t
        elif (score_remain <= 40 and score_remain%2==0):
            prob_notbust_nt += prob_doublescore_dic_nt[score_remain]
            prob_notbust_t += prob_doublescore_dic_t[score_remain]
        ##
        prob_notbust_nt = np.minimum(np.maximum(prob_notbust_nt, 0),1)
        prob_notbust_dic_nt[score_max] = prob_notbust_nt
        prob_bust_dic_nt[score_max] = 1 - prob_notbust_dic_nt[score_max]
        prob_notbust_t = np.minimum(np.maximum(prob_notbust_t, 0),1)
        prob_notbust_dic_t[score_max] = prob_notbust_t
        prob_bust_dic_t[score_max] = 1 - prob_notbust_dic_t[score_max]
    
    return num_aiming_location, prob_normalscore_nt, prob_doublescore_dic_nt, prob_DB_nt, prob_bust_dic_nt, prob_notbust_dic_nt, prob_normalscore_t, prob_doublescore_dic_t, prob_DB_t, prob_bust_dic_t, prob_notbust_dic_t


def solve_turn_transit_probability_fast_token(score_state, state_action, available_tokens, prob_grid_normalscore_nt, prob_grid_doublescore_nt, prob_grid_bullscore_nt, prob_bust_dic_nt,prob_grid_normalscore_t, prob_grid_doublescore_t, prob_grid_bullscore_t, prob_bust_dic_t):

    parent_probability = np.ones((1,1))

    # Establish number of tokens that must be considered in turn  
    max_tokens_in_turn = min(available_tokens,3) # either 3 (use token for every throw in turn) or the total number of available tokens if < 3 
    max_tokens_plus_one = max_tokens_in_turn + 1 # used for indexing purposes

    # Initialize data structures to store final probabilities
    result_dict = {}
    result_dict['finish'] = 0 # probability of finishing the game
    result_dict['bust'] = np.zeros(max_tokens_plus_one) # probability of going bust while using n tokens
    prob_transit_len = min(score_state-2, fb.maxhitscore*(3)) + 1 # length of feasible transition array 
    result_dict['score'] = np.zeros((max_tokens_plus_one,prob_transit_len)) # matrix of TPs; prob of transitioning to [t,s] where t is tokens used and s is score state


    for rt in [3,2,1]:

        # Max score gained in turn; bounded by score state if you can go bust. Otherwise will be 60,120,180 for rt=3,2,1 respectively. 
        max_score_gained = min(score_state-2, fb.maxhitscore*(4-rt))

        # Add a one for indexing purposes 
        max_score_gained_index = max_score_gained + 1

        # Max score gained by throw 
        max_throw_score = min(score_state-2, fb.maxhitscore)
        max_throw_score_index = max_throw_score + 1

        # Token sizing 
        token_index = min(available_tokens,4-rt) + 1

        # Initialize child probability object which will contain each state that can be transitioned to, with value equal to transition prob
        child_probability = np.zeros((token_index,max_score_gained_index))

        # this fills in the parent probability 
        #prob_normalscore_transit = prob_normalscore[state_action[rt][0:this_throw_state_len]]*prob_this_throw_state.reshape((this_throw_state_len,1))
        
        for tokens_used in range(parent_probability.shape[0]):

            for score_gained in range(parent_probability.shape[1]):

                ## skip infeasible state
                if not fb.state_feasible_array[rt, score_gained]:
                    continue   

                ## skip if zero probability of being in parent state in the first place to save time
                if parent_probability[tokens_used][score_gained] < 0.000000000000001:
                    continue 

                ## get policy action at this state
                remaining_tokens = available_tokens-tokens_used
                action_index = state_action[rt][remaining_tokens][score_gained]
                is_token = (action_index >= imdp.throw_num)
                bool_token = 1 if is_token else 0 

                if ((is_token)&((max_tokens_in_turn-tokens_used)>0)): 
                    prob_normalscore = prob_grid_normalscore_t
                    prob_bullscore = prob_grid_bullscore_t
                    prob_doublescore = prob_grid_doublescore_t
                    prob_bust_dic = prob_bust_dic_t
                else: 
                    prob_normalscore = prob_grid_normalscore_nt
                    prob_bullscore = prob_grid_bullscore_nt
                    prob_doublescore = prob_grid_doublescore_nt
                    prob_bust_dic = prob_bust_dic_nt

                # Probabilitiy of being in this state of the turn (i.e. specific number of tokens used and score gained) given policy
                prob_this_state = parent_probability[tokens_used][score_gained]

                # Probability of every outcome of current throw (2x61); either 0 or 1 tokens used, score_gained of 0 to 60
                throw_transit_probability = np.zeros((min(remaining_tokens+1,2),max_throw_score_index))

                # Add transition probabilities depending on whether a token was used or not; multiply by probability of this state 
                if available_tokens ==0: 
                    throw_transit_probability += prob_normalscore[action_index][:max_throw_score_index] * prob_this_state 
                else: 
                    throw_transit_probability[bool_token] += prob_normalscore[action_index][:max_throw_score_index] * prob_this_state 

                # Get the largest score that can be made without busting 
                score_remain = score_state - score_gained
                score_max = min(score_remain-2, 60)
                score_max_plus1 = score_max + 1
            
                ## Transit to next throw or turn with normal scores    
                if available_tokens ==0: 
                    child_probability[0,score_gained:score_gained+score_max_plus1] += throw_transit_probability[0,0:score_max_plus1]
                else:    
                    child_probability[tokens_used:tokens_used+2,score_gained:score_gained+score_max_plus1] += throw_transit_probability[:,0:score_max_plus1]
                
                ## game can not bust or end when score_max = 60, i.e.,  prob_notbust = 1
                if (score_max < 60):
                    ## transit to the end of game
                    if (score_remain == fb.score_DB):
                        result_dict['finish'] += prob_bullscore[action_index, 1]*prob_this_state
                    elif (score_remain <= 40 and score_remain%2==0):
                        doublescore_index = (score_remain//2) - 1
                        result_dict['finish'] += prob_doublescore[action_index, doublescore_index]*prob_this_state
                    else:
                        pass

                    #transit to bust
                    result_dict['bust'][tokens_used+bool_token] += prob_bust_dic[score_max][action_index]*prob_this_state

        parent_probability = child_probability
        #print(parent_probability)

    result_dict['score'] = parent_probability

    return result_dict

def solve_policy_transit_probability(tokens,policy_action_index_dic, prob_grid_normalscore_nt, prob_grid_doublescore_nt, prob_grid_bullscore_nt, prob_bust_dic_nt,prob_grid_normalscore_t, prob_grid_doublescore_t, prob_grid_bullscore_t, prob_bust_dic_t):
    """
    For each turn, solve the state transition probability for a specified aiming policy
    
    Args: 
        policy_action_index_dic: a dict of aiming locations (actions in the policy) for each state (s,i,u) of each turn s=2,...,501
        prob_normalscore, prob_doublescore, prob_bullscore: the skill model 
    
    Returns: A dict
    """  
    
    prob_policy_transit_dict = {}
    t1 = time.time()

    for tok in range(0,tokens+1):
        
        prob_policy_transit_dict[tok] = {}

        for score_state in range(2,502):

            prob_policy_transit_dict[tok][score_state] = solve_turn_transit_probability_fast_token(score_state, policy_action_index_dic[score_state], tok, prob_grid_normalscore_nt, prob_grid_doublescore_nt, prob_grid_bullscore_nt, prob_bust_dic_nt,prob_grid_normalscore_t, prob_grid_doublescore_t, prob_grid_bullscore_t, prob_bust_dic_t)
            #prob_policy_transit_dict[score_state] = solve_turn_transit_probability(score_state, policy_action_index_dic[score_state], prob_grid_normalscore, prob_grid_doublescore, prob_grid_bullscore)

    t2 = time.time()
    print('solve prob_policy_transit in {} seconds'.format(t2-t1))
    
    return prob_policy_transit_dict

#%%
## single player game without the turn feature
def solve_dp_noturn_tokens(aiming_grid, prob_grid_normalscore, prob_grid_normalscore_t, tokens = 0, prob_grid_doublescore=None, prob_grid_bullscore=None, prob_grid_doublescore_dic=None, prob_grid_doublescore_t=None, prob_grid_bullscore_t=None, prob_grid_doublescore_dic_t=None):
    """
    Solve the single player game without the turn feature. Find the optimal policy to minimize the expected number of throws for reaching zero score. 
    Args: 
        the action set and the hitting probability associated with the skill model
    
    Returns: 
        optimal_value[score_state]: the expected number of throws for reaching zero from score_state=2,...,501.
        optimal_action_index[score_state]: the index of the aiming location used for score_state=2,...,501.
    """

    max_token_index = tokens + 1

    num_aiming_location = aiming_grid.shape[0]
    prob_normalscore_1tosmax_dic = {}
    prob_normalscore_1tosmaxsum_dic = {}
    prob_normalscore_1tosmax_dic_t = {}
    prob_normalscore_1tosmaxsum_dic_t = {}

    # No tokens
    for score_max in range(0,61):
        score_max_plus1 = score_max + 1 
        prob_normalscore_1tosmax_dic[score_max] = np.array(prob_grid_normalscore[:,1:score_max_plus1])
        prob_normalscore_1tosmaxsum_dic[score_max] = prob_normalscore_1tosmax_dic[score_max].sum(axis=1)
    if prob_grid_doublescore_dic is None:
        prob_doublescore_dic = {}
        for doublescore_index in range(20):
            doublescore = 2*(doublescore_index+1)
            prob_doublescore_dic[doublescore] = np.array(prob_grid_doublescore[:,doublescore_index])
    else:
        prob_doublescore_dic = prob_grid_doublescore_dic
    prob_DB = np.array(prob_grid_bullscore[:,1])

    # Tokens
    for score_max in range(0,61):
        score_max_plus1 = score_max + 1 
        prob_normalscore_1tosmax_dic_t[score_max] = np.array(prob_grid_normalscore_t[:,1:score_max_plus1])
        prob_normalscore_1tosmaxsum_dic_t[score_max] = prob_normalscore_1tosmax_dic_t[score_max].sum(axis=1)
    if prob_grid_doublescore_dic_t is None:
        prob_doublescore_dic_t = {}
        for doublescore_index in range(20):
            doublescore = 2*(doublescore_index+1)
            prob_doublescore_dic_t[doublescore] = np.array(prob_grid_doublescore_t[:,doublescore_index])
    else:
        prob_doublescore_dic_t = prob_grid_doublescore_dic_t
    prob_DB_t = np.array(prob_grid_bullscore_t[:,1])

    ## possible state: s = 0,1(not possible),2,...,501
    optimal_value = np.zeros((max_token_index,502))
    #optimal_value[1] = np.nan
    optimal_action_index = np.zeros((max_token_index,502), np.int32)
    optimal_action_index[:,0] = -1
    optimal_action_index[:,1] = -1

    for t in range(0,max_token_index):

        if t==0: 

            for score_state in range(2,502):            
                ## use matrix operation to search all aiming locations
                
                ## transit to less score state    
                ## s1 = min(score_state-2, 60)
                ## p[z=1]*v[score_state-1] + p[z=2]*v[score_state-2] + ... + p[z=s1]*v[score_state-s1]
                score_max = min(score_state-2, 60)
                score_max_plus1 = score_max + 1 

                ## transit to next state
                num_tothrow = 1.0 + prob_normalscore_1tosmax_dic[score_max].dot(optimal_value[0,score_state-1:score_state-score_max-1:-1])
                ## probability of transition to state other than s itself
                prob_otherstate = prob_normalscore_1tosmaxsum_dic[score_max]
                
                ## transit to the end of game
                if (score_state == fb.score_DB): ## hit double bull
                    prob_otherstate += prob_DB
                elif (score_state <= 40 and score_state%2==0): ## hit double
                    prob_otherstate += prob_doublescore_dic[score_state]
                else: ## game does not end
                    pass
                
                ## expected number of throw for all aiming locations
                prob_otherstate = np.maximum(prob_otherstate, 0)
                num_tothrow = num_tothrow / prob_otherstate
                                    
                ## searching
                optimal_value[0,score_state] = num_tothrow.min()
                optimal_action_index[0,score_state] = num_tothrow.argmin()
        
        else:

            for score_state in range(2,502):            
                ## use matrix operation to search all aiming locations
                
                ## transit to less score state    
                ## s1 = min(score_state-2, 60)
                ## p[z=1]*v[score_state-1] + p[z=2]*v[score_state-2] + ... + p[z=s1]*v[score_state-s1]
                score_max = min(score_state-2, 60)
                score_max_plus1 = score_max + 1 

                ## transit to next state
                num_tothrow = 1.0 
                # tp_nt * v_t <-- save token and make throw
                num_tothrow+= prob_normalscore_1tosmax_dic[score_max].dot(optimal_value[t,score_state-1:score_state-score_max-1:-1])
                # tp_t * v_nt <-- use token 
                num_tothrow+= prob_normalscore_1tosmax_dic_t[score_max].dot(optimal_value[t-1,score_state-1:score_state-score_max-1:-1])

                ## probability of transition to state other than s itself
                prob_otherstate = prob_normalscore_1tosmaxsum_dic[score_max] + prob_normalscore_1tosmaxsum_dic_t[score_max]
                
                ## transit to the end of game
                if (score_state == fb.score_DB): ## hit double bull
                    prob_otherstate += prob_DB + prob_DB_t
                elif (score_state <= 40 and score_state%2==0): ## hit double
                    prob_otherstate += prob_doublescore_dic[score_state]
                    prob_otherstate += prob_doublescore_dic_t[score_state]
                else: ## game does not end
                    pass
                
                ## expected number of throw for all aiming locations
                prob_otherstate = np.maximum(prob_otherstate, 0)
                num_tothrow = num_tothrow / prob_otherstate
                                    
                ## searching
                optimal_value[t,score_state] = num_tothrow.min()
                optimal_action_index[t,score_state] = num_tothrow.argmin()
        
    return [optimal_value, optimal_action_index]


def solve_dp_turn_tokens(tokens, aiming_grid, prob_grid_normalscore_nt, prob_grid_singlescore_nt, prob_grid_doublescore_nt, prob_grid_triplescore_nt, prob_grid_bullscore_nt,prob_grid_normalscore_t, prob_grid_singlescore_t, prob_grid_doublescore_t, prob_grid_triplescore_t, prob_grid_bullscore_t):

    tokens_plus_one = tokens + 1

    ## Initialize probability objects
    num_aiming_location, prob_normalscore_nt, prob_doublescore_dic_nt, prob_DB_nt, prob_bust_dic_nt, prob_notbust_dic_nt, prob_normalscore_t, prob_doublescore_dic_t, prob_DB_t, prob_bust_dic_t, prob_notbust_dic_t = init_probabilities( aiming_grid, prob_grid_normalscore_nt, prob_grid_doublescore_nt, prob_grid_bullscore_nt, prob_grid_normalscore_t, prob_grid_doublescore_t, prob_grid_bullscore_t)    

    # Initialize joint probability notbust dictionary 
    prob_notbust_dic = {}
    for score_max in prob_notbust_dic_t.keys():
        prob_notbust_dic[score_max] = np.zeros(prob_notbust_dic_t[0].shape[0]) # 791 
        prob_notbust_dic[score_max][:imdp.throw_num] += prob_notbust_dic_nt[score_max][:imdp.throw_num] 
        prob_notbust_dic[score_max][imdp.throw_num:] += prob_notbust_dic_t[score_max][imdp.throw_num:]

    # Intialize joint probability bust dictionary 
    prob_bust_dic = {}
    for score_max in prob_notbust_dic_t.keys():
        prob_bust_dic[score_max] = np.zeros(prob_bust_dic_t[0].shape[0]) # 791 
        prob_bust_dic[score_max][:imdp.throw_num] += prob_bust_dic_nt[score_max][:imdp.throw_num] 
        prob_bust_dic[score_max][imdp.throw_num:] += prob_bust_dic_t[score_max][imdp.throw_num:]


    prob_normalscore_tensor_nt = torch.from_numpy(prob_normalscore_nt)
    prob_normalscore_tensor_t = torch.from_numpy(prob_normalscore_t)

    # prob_doublescore_dic = prob_doublescore_dic_nt
    # prob_DB = prob_DB_nt
    # prob_bust_dic = prob_bust_dic_nt
    # prob_notbust_dic = prob_notbust_dic_nt

    iteration_round_limit = 20

    optimal_value_rt3 = np.zeros((tokens_plus_one,502)) #vector: optimal value for the beginning state of each turn (rt=3)
    optimal_value_dic = {} ## first key: score=0,2,...,501, second key: remaining throws=3,2,1
    optimal_action_index_dic = {}
    num_iteration_record = np.zeros(502, dtype=np.int32)

    state_len_vector = np.zeros(4, dtype=np.int32)
    state_value  = [None]  ## optimal value (expected # of turns to finish the game) for each state in the current playing turn
    state_action = [None]  ## aimming locations for for each state in the current playing turn
    action_diff  = [None]
    value_relerror = np.zeros(4)

    for rt in [1,2,3]:
        ## for rt=3: possible score_gained = 0
        ## for rt=2: possible score_gained = 0,1,...,60
        ## for rt=1: possible score_gained = 0,1,...,120
        this_throw_state_len = fb.maxhitscore*(3-rt) + 1
        state_value.append(np.ones((tokens_plus_one,this_throw_state_len))*fb.largenumber)
        state_action.append(np.ones((tokens_plus_one,this_throw_state_len), np.int32)*fb.infeasible_marker)
        action_diff.append(np.ones((tokens_plus_one,this_throw_state_len)))
    state_value_update = ft.copy_numberarray_container(state_value)
    state_action_update = ft.copy_numberarray_container(state_action)

    ## use no_turn policy as the initial policy
    #[noturn_optimal_value, noturn_optimal_action_index] = solve_dp_noturn(aiming_grid, prob_grid_normalscore, prob_grid_doublescore, prob_grid_bullscore)
    [noturn_optimal_value, noturn_optimal_action_index] = solve_dp_noturn_tokens(aiming_grid, prob_grid_normalscore_nt, prob_grid_normalscore_t, tokens = tokens, prob_grid_doublescore=prob_grid_doublescore_nt, prob_grid_bullscore=prob_grid_bullscore_nt, prob_grid_doublescore_dic=None, prob_grid_doublescore_t=prob_grid_doublescore_t, prob_grid_bullscore_t=prob_grid_bullscore_t, prob_grid_doublescore_dic_t=None)

    t1 = time.time()
    for score_state in range(2,502):

        for tok in range(tokens_plus_one):
            
            ## initialization 
            for rt in [1,2,3]:
            
                ## for rt=3: score_gained = 0
                ## for rt=2: score_gained = 0,1,...,min(s-2,60)
                ## for rt=1: score_gained = 0,1,...,min(s-2,120)
                this_throw_state_len = min(score_state-2, fb.maxhitscore*(3-rt)) + 1
                state_len_vector[rt] = this_throw_state_len
                        
                ## initialize the starting policy: 
                ## use no_turn action in (s, i, u=0)
                ## use turn action (s-1, i, u-1) in (s, i, u!=0) if (s-1, i, u-1) is feasible state
                state_action[rt][tok][0] = noturn_optimal_action_index[tok][score_state]            
                for score_gained in range(1,this_throw_state_len):                
                    if fb.state_feasible_array[rt, score_gained]:  ## if True
                        if fb.state_feasible_array[rt, score_gained-1]:
                            state_action[rt][tok][score_gained] = optimal_action_index_dic[score_state-1][rt][tok][score_gained-1]
                        else:                        
                            state_action[rt][tok][score_gained] = noturn_optimal_action_index[tok][score_state-score_gained]
                    else:
                        state_action[rt][tok][score_gained] = fb.infeasible_marker
        
        # for tok in range(tokens_plus_one):

            # policy iteration
            for round_index in range(iteration_round_limit):

                ## --------------------------------------------- ##
                ##                 POLICY EVALUATION
                ## --------------------------------------------- ##
                rt = 3
                score_gained = 0
                score_max_turn = min(score_state-2, 3*fb.maxhitscore)

                # Get the transit probabilities 
                prob_turn_transit = solve_turn_transit_probability_fast_token(score_state, state_action, tok, prob_grid_normalscore_nt, prob_grid_doublescore_nt, prob_grid_bullscore_nt, prob_bust_dic_nt,prob_grid_normalscore_t, prob_grid_doublescore_t, prob_grid_bullscore_t, prob_bust_dic_t)
                
                tokens_max_turn = min(prob_turn_transit['bust'].shape[0]-1,tok)

                # Make updates below 
                prob_turn_zeroscore = prob_turn_transit['bust'][0] + prob_turn_transit['score'][0,0]

                new_value_rt3 = 1 

                # Don't use a token 
                new_value_rt3 += np.dot(prob_turn_transit['score'][0,1:], optimal_value_rt3[tok,score_state-1:score_state-score_max_turn-1:-1])
                
                # Use at least one token 
                if tok > 0: 

                    # Go to subsequent score after using at least one token
                    for i in range(0,tokens_max_turn):
                        #new_value_rt3 += np.dot(prob_turn_transit['score'][0:tokens_max_turn+1,:][i], optimal_value_rt3[tok-tokens_max_turn:tok+1,score_state:score_state-score_max_turn-1:-1][i])

                        new_value_rt3 += np.dot(np.flip(prob_turn_transit['score'][0:tokens_max_turn+1,:],axis=0)[i], optimal_value_rt3[tok-tokens_max_turn:tok+1,score_state:score_state-score_max_turn-1:-1][i])
                
                    # Go bust after using at least one token 
                    new_value_rt3 += np.dot(np.flip(prob_turn_transit['bust'][1:tokens_max_turn+1]) , optimal_value_rt3[tok-tokens_max_turn+1:tok+1,score_state])

                # Normalize over probability of moving to a new state 
                new_value_rt3 = new_value_rt3 / (1-prob_turn_zeroscore)
                new_value_rt3

                state_value_update[rt][tok][score_gained] = new_value_rt3
                optimal_value_rt3[tok][score_state] = new_value_rt3
                #print('evaluate rt3 value= {}'.format(new_value_rt3)

                ## --------------------------------------------- ##
                ##                 POLICY IMPROVEMENT
                ## --------------------------------------------- ## 

                for rt in [1,2,3]: 

                    this_throw_state_len = state_len_vector[rt]
                    state_notbust_len =  max(min(score_state-61, this_throw_state_len),0)
                    token_index = min(2,tok+1)

                    ## CASE 1: state which can not bust.  score_state-score_gained>=62 
                    if (state_notbust_len > 0):
                        # One throw remaining and first round of iteration 
                        if (rt==1 and round_index==0):
                            ## combine all non-bust states together 
                            state_notbust_update_index = state_notbust_len                    
                            next_state_value_array_nt = np.zeros((61, state_notbust_len))   
                            next_state_value_array_t = np.zeros((61, state_notbust_len))             
                            for score_gained in range(state_notbust_len):
                                ## skip infeasible state
                                if not fb.state_feasible_array[rt, score_gained]:
                                    continue
                                score_remain = score_state - score_gained
                                score_max = 60 ## always 60 here
                                score_max_plus1 = score_max + 1

                                new_state_vals_nt = optimal_value_rt3[tok]
                                next_state_value_array_nt[:,score_gained] = new_state_vals_nt[score_remain:score_remain-score_max_plus1:-1]

                                if tok > 0: 
                                    new_state_vals_t = optimal_value_rt3[tok-1]
                                    next_state_value_array_t[:,score_gained] = new_state_vals_t[score_remain:score_remain-score_max_plus1:-1]
                                    
                        # Two throws remaining and first round of iteration 
                        elif (rt==2 and (round_index==0 or score_state<182)):
                            ## combine all non-bust states together 
                            state_notbust_update_index = state_notbust_len
                            next_state_value_array_nt = np.zeros((61, state_notbust_len))      
                            next_state_value_array_t = np.zeros((61, state_notbust_len))                      
                            for score_gained in range(state_notbust_len):
                                ## skip infeasible state
                                if not fb.state_feasible_array[rt, score_gained]:
                                    continue
                                score_remain = score_state - score_gained
                                score_max = 60 ## always 60 here
                                score_max_plus1 = score_max + 1

                                new_state_vals_nt = state_value_update[rt-1][tok]
                                next_state_value_array_nt[:,score_gained] = new_state_vals_nt[score_gained:score_gained+score_max_plus1]

                                if tok > 0: 
                                    new_state_vals_t = state_value_update[rt-1][tok-1]
                                    next_state_value_array_t[:,score_gained] = new_state_vals_t[score_gained:score_gained+score_max_plus1]

                                    
                        # Three throws remaining 
                        # OR two throws remaining past first round of iteration and score state is greater than 182 
                        # OR one throw remaining past first round of iteration  
                        else: ##(rt==1 and round_index>0) or (rt==2 and round_index>0 and score_state>=182) or (rt==3)
                            ## only update state of score_gained = 0
                            state_notbust_update_index = 1
                            next_state_value_array_nt= np.zeros((61))
                            next_state_value_array_t= np.zeros((61))
                            score_gained = 0
                            score_remain = score_state - score_gained
                            score_max = 60 ## always 60 here
                            score_max_plus1 = score_max + 1                    
                            ## make a copy
                            if (rt > 1):

                                new_state_vals_nt = state_value_update[rt-1][tok]
                                next_state_value_array_nt[:] = new_state_vals_nt[score_gained:score_gained+score_max_plus1]
                                
                                if tok > 0:
                                    new_state_vals_t = state_value_update[rt-1][tok-1]
                                    next_state_value_array_t[:] = new_state_vals_t[score_gained:score_gained+score_max_plus1]

                            ## transit to next turn when rt=1
                            else:
                                new_state_vals_nt = optimal_value_rt3[tok]
                                next_state_value_array_nt[:] = new_state_vals_nt[score_remain:score_remain-score_max_plus1:-1]
                                
                                if tok > 0: 
                                    new_state_vals_t = optimal_value_rt3[tok-1]
                                    next_state_value_array_t[:] = new_state_vals_t[score_remain:score_remain-score_max_plus1:-1]
                                    
        
                        ## matrix product to compute all together
                        next_state_value_tensor_nt = torch.from_numpy(next_state_value_array_nt)
                        next_state_value_tensor_t = torch.from_numpy(next_state_value_array_t)
                        ## transit to next throw in the same turn when rt=3,2
                        if (rt > 1): 

                            # add one to indicate end of turn 
                            try: 
                                num_turns_tensor = torch.zeros((len(aiming_grid),next_state_value_array_nt.shape[1]))
                            except:
                                num_turns_tensor = torch.zeros(len(aiming_grid))
                            
                            # if we have tokens 
                            if tok > 0:  

                                # use no token probabilities for no token actions 
                                num_turns_tensor[:imdp.throw_num] += prob_normalscore_tensor_nt[:imdp.throw_num].matmul(next_state_value_tensor_nt)
                                # use token probabilities for token actions 
                                num_turns_tensor[imdp.throw_num:len(aiming_grid)] += prob_normalscore_tensor_t[imdp.throw_num:].matmul(next_state_value_tensor_t)

                            # if we don't have tokens   
                            else:
                                # add no token expectation --> index is 1 because we keep the same # of tokens       
                                num_turns_tensor = prob_normalscore_tensor_nt.matmul(next_state_value_tensor_nt)
                            
                        ## transit to next turn when rt=1
                        else:

                            # add one to indicate end of turn 
                            try: 
                                num_turns_tensor = torch.ones((len(aiming_grid),next_state_value_array_nt.shape[1]))
                            except:
                                num_turns_tensor = torch.ones(len(aiming_grid))
                        
                            # if we have tokens 
                            if tok > 0: 
                                # use no token probabilities for no token actions
                                num_turns_tensor[:imdp.throw_num] += prob_normalscore_tensor_nt[:imdp.throw_num].matmul(next_state_value_tensor_nt)
                                
                                # use token probabilities for token actions 
                                num_turns_tensor[imdp.throw_num:] += prob_normalscore_tensor_t[imdp.throw_num:].matmul(next_state_value_tensor_t)
                                #num_turns_tensor[imdp.throw_num:791] += prob_normalscore_tensor_t[imdp.throw_num:].matmul(next_state_value_tensor_t)
                            else: 
                                # add no token expectation  
                                num_turns_tensor += prob_normalscore_tensor_nt.matmul(next_state_value_tensor_nt)

                        ## searching
                        #temp1 = num_turns_tensor.min(axis=0) 
                        if tok == 0: 
                            temp1 = num_turns_tensor[:imdp.throw_num].min(axis=0)
                        else: 
                            temp1 = num_turns_tensor.min(axis=0)               
                        state_action_update[rt][tok][0:state_notbust_update_index] = temp1.indices.numpy()
                        state_value_update[rt][tok][0:state_notbust_update_index] =  temp1.values.numpy() 

                    ## CASE 2: state which possibly bust.  score_state-score_gained<62 
                    if (state_notbust_len < this_throw_state_len):
                        ## combine all bust states together 
                        state_bust_len = this_throw_state_len - state_notbust_len
                        next_state_value_array_nt = np.zeros((61, state_bust_len))
                        next_state_value_array_t = np.zeros((61, state_bust_len))
                        for score_gained in range(state_notbust_len, this_throw_state_len):
                            ## skip infeasible state
                            if not fb.state_feasible_array[rt, score_gained]:
                                continue
                            score_remain = score_state - score_gained
                            #score_max = min(score_remain-2, 60)
                            score_max = score_remain-2 ## less than 60 here
                            score_max_plus1 = score_max + 1
                            score_gained_index = score_gained - state_notbust_len ## index off set
                            if (rt > 1):

                                new_state_vals_nt = state_value_update[rt-1][tok]
                                next_state_value_array_nt[0:score_max_plus1,score_gained_index] = new_state_vals_nt[score_gained:score_gained+score_max_plus1]

                                if tok > 0: 
                                    new_state_vals_t = state_value_update[rt-1][tok-1]
                                    next_state_value_array_t[0:score_max_plus1,score_gained_index] = new_state_vals_t[score_gained:score_gained+score_max_plus1]

                            ## transit to next turn when rt=1
                            else:

                                new_state_vals_nt = optimal_value_rt3[tok]
                                next_state_value_array_nt[0:score_max_plus1,score_gained_index] = new_state_vals_nt[score_remain:score_remain-score_max_plus1:-1]

                                
                                if tok > 0: 
                                    new_state_vals_t = optimal_value_rt3[tok-1]
                                    next_state_value_array_t[0:score_max_plus1,score_gained_index] = new_state_vals_t[score_remain:score_remain-score_max_plus1:-1]
                                    
                        next_state_value_tensor_nt = torch.from_numpy(next_state_value_array_nt)
                        next_state_value_tensor_t = torch.from_numpy(next_state_value_array_t)
                        
                        ## transit to next throw in the same turn when rt=3,2
                        if (rt > 1):  

                            # initialized
                            try: 
                                num_turns_tensor = torch.zeros((len(aiming_grid),next_state_value_array_nt.shape[1]))
                            except:
                                num_turns_tensor = torch.zeros(len(aiming_grid))
                        
                            # if we have tokens 
                            if tok > 0: 
                                # use no token probabilities for no token actions
                                num_turns_tensor[:imdp.throw_num] += prob_normalscore_tensor_nt[:imdp.throw_num].matmul(next_state_value_tensor_nt)
                                
                                # use token probabilities for token actions 
                                num_turns_tensor[imdp.throw_num:] += prob_normalscore_tensor_t[imdp.throw_num:].matmul(next_state_value_tensor_t)

                                #num_turns_tensor[imdp.throw_num:791] += prob_normalscore_tensor_t[imdp.throw_num:].matmul(next_state_value_tensor_t)
                            else: 
                                # add no token expectation  
                                num_turns_tensor += prob_normalscore_tensor_nt.matmul(next_state_value_tensor_nt)

                                            
                        ## transit to next turn when rt=1
                        else:

                            # add one to indicate end of turn 
                            try: 
                                num_turns_tensor = torch.ones((len(aiming_grid),next_state_value_array_nt.shape[1]))
                            except:
                                num_turns_tensor = torch.ones(len(aiming_grid))
                        
                            # if we have tokens 
                            if tok > 0: 
                                # use no token probabilities for no token actions
                                num_turns_tensor[:imdp.throw_num] += prob_normalscore_tensor_nt[:imdp.throw_num].matmul(next_state_value_tensor_nt)
                                
                                # use token probabilities for token actions 
                                num_turns_tensor[imdp.throw_num:] += prob_normalscore_tensor_t[imdp.throw_num:].matmul(next_state_value_tensor_t)

                                #num_turns_tensor[imdp.throw_num:791] += prob_normalscore_tensor_t[imdp.throw_num:].matmul(next_state_value_tensor_t)
                            else: 
                                # add no token expectation  
                                num_turns_tensor += prob_normalscore_tensor_nt.matmul(next_state_value_tensor_nt)

                            # # add one to indicate end of turn 
                            # num_turns_tensor = 1 
                            # # add no token expectation  
                            # num_turns_tensor += prob_normalscore_tensor_nt.matmul(next_state_value_tensor[0])
                            
                            # if tok > 0: 
                            #     # add token expectation 
                            #     num_turns_tensor += prob_normalscore_tensor_t.matmul(next_state_value_tensor[1])                                                             
        
                        ## consider bust/finishing for each bust state seperately 
                        num_turns_array = num_turns_tensor.numpy() 
                        ## for every score gained where you go bust                
                        for score_gained in range(state_notbust_len, this_throw_state_len):
                            ## skip infeasible state
                            if not fb.state_feasible_array[rt, score_gained]:
                                continue
                            score_remain = score_state - score_gained
                            #score_max = min(score_remain-2, 60)
                            score_max = score_remain-2 ## less than 60 here
                            score_max_plus1 = score_max + 1
                            score_gained_index = score_gained - state_notbust_len
                            
                            ## transit to the end of game
                            if (rt > 1):
                                if (score_remain == fb.score_DB):                        
                                    num_turns_array[:imdp.throw_num,score_gained_index] += prob_DB_nt[:imdp.throw_num]
                                    num_turns_array[imdp.throw_num:,score_gained_index] += prob_DB_t[imdp.throw_num:]
                                elif (score_remain <= 40 and score_remain%2==0):
                                    num_turns_array[:imdp.throw_num,score_gained_index] += prob_doublescore_dic_nt[score_remain][:imdp.throw_num]
                                    num_turns_array[imdp.throw_num:,score_gained_index] += prob_doublescore_dic_t[score_remain][imdp.throw_num:]
                                else:
                                    pass
        
                            ## transit to bust
                            if (rt==3):
                                # In this case you really are staying in the same state becuase it's the first throw of the turn
                                # Unless you go bust with the current token policy (should never happen, but need to include to avoid choosing those actions)
                                if tok > 0: 
                                    
                                    # For no token, add the probbability of staying the same with nt 
                                    num_turns_array[:imdp.throw_num,score_gained_index] += prob_bust_dic_nt[score_max][:imdp.throw_num]
                                    ## solve an equation other than using the policy evaluation value (s,i=3,u=0)
                                    num_turns_array[:imdp.throw_num,score_gained_index] = num_turns_array[:imdp.throw_num,score_gained_index] / prob_notbust_dic_nt[score_max][:imdp.throw_num]
                                    
                                    # For token, add the probability of going to the new rt3 state (1 should already be included)
                                    # TODO check that this makes sense - not sure about the +1 here 
                                    num_turns_array[imdp.throw_num:,score_gained_index] += prob_bust_dic_t[score_max][imdp.throw_num:]*(1+optimal_value_rt3[tok-1,score_state])  ## 1 turn is already counted before
                                    ## solve an equation other than using the policy evaluation value (s,i=3,u=0)
                                    num_turns_array[imdp.throw_num:,score_gained_index] = num_turns_array[imdp.throw_num:,score_gained_index] / prob_notbust_dic_t[score_max][imdp.throw_num:]
                                    
                                    
                                else: 
                                    # If no token, use this logic with nt 
                                    num_turns_array[:,score_gained_index] += prob_bust_dic_nt[score_max]
                                    ## solve an equation other than using the policy evaluation value (s,i=3,u=0)
                                    num_turns_array[:,score_gained_index] = num_turns_array[:,score_gained_index] / prob_notbust_dic_nt[score_max] 
                                    
                            elif (rt==2):
                                num_turns_array[:imdp.throw_num,score_gained_index] += prob_bust_dic_nt[score_max][:imdp.throw_num]*(1+new_value_rt3)
                                num_turns_array[imdp.throw_num:,score_gained_index] += prob_bust_dic_t[score_max][imdp.throw_num:]*(1+optimal_value_rt3[tok-1,score_state])
                            else:
                                # TODO check that the +1 makes sense here 
                                num_turns_array[:imdp.throw_num,score_gained_index] += prob_bust_dic_nt[score_max][:imdp.throw_num]*(new_value_rt3)  ## 1 turn is already counted before
                                num_turns_array[imdp.throw_num:,score_gained_index] += prob_bust_dic_t[score_max][imdp.throw_num:]*(1+optimal_value_rt3[tok-1,score_state])
                        
                        ## searching
                        if tok == 0: 
                            temp1 = num_turns_tensor[:imdp.throw_num].min(axis=0)
                        else: 
                            temp1 = num_turns_tensor.min(axis=0)
                        state_action_update[rt][tok][state_notbust_len:this_throw_state_len] = temp1.indices.numpy()
                        state_value_update[rt][tok][state_notbust_len:this_throw_state_len] =  temp1.values.numpy()                
        
                    #### finish rt=1,2,3. check improvement
                    action_diff[rt][tok][:] = np.abs(state_action_update[rt][tok] - state_action[rt][tok])                                
                    value_relerror[rt] = np.abs((state_value_update[rt] - state_value[rt])/state_value_update[rt]).max()
                    state_action[rt][tok][:] = state_action_update[rt][tok][:]
                    state_value[rt][tok][:] = state_value_update[rt][tok][:]
        
                max_action_diff = max([action_diff[1].max(), action_diff[2].max(), action_diff[3].max()])
                max_value_relerror = value_relerror.max()            
                
                if (max_action_diff < 1):
                #if max_value_relerror < iteration_relerror_limit:
                    num_iteration_record[score_state] = round_index + 1
                    break


            for rt in [1,2,3]:
                state_value_update[rt][tok][fb.state_infeasible[rt]] = fb.largenumber
                state_action_update[rt][tok][fb.state_infeasible[rt]] = fb.infeasible_marker
            optimal_action_index_dic[score_state] = ft.copy_numberarray_container(state_action_update)
            optimal_value_dic[score_state] = ft.copy_numberarray_container(state_value_update, new_dtype=fb.result_float_dytpe)
            optimal_value_rt3[tok][score_state] = state_value[3][tok][0]

    prob_scorestate_transit = {}    
    #prob_scorestate_transit =  fep.solve_policy_transit_probability(optimal_action_index_dic, prob_grid_normalscore, prob_grid_doublescore, prob_grid_bullscore)
    prob_scorestate_transit = solve_policy_transit_probability(tokens, optimal_action_index_dic, prob_grid_normalscore_nt, prob_grid_doublescore_nt, prob_grid_bullscore_nt, prob_bust_dic_nt,prob_grid_normalscore_t, prob_grid_doublescore_t, prob_grid_bullscore_t, prob_bust_dic_t)
    t2 = time.time()
    print('solve dp_turn_policyiter in {} seconds'.format(t2-t1))

    print(optimal_value_rt3)
    result_dic = {'optimal_value_dic':optimal_value_dic, 'optimal_action_index_dic':optimal_action_index_dic, 'optimal_value_rt3':optimal_value_rt3, 'prob_scorestate_transit':prob_scorestate_transit}

    return result_dic 


def solve_singlegame_token(name_pa, epsilon, tokens=9, data_parameter_dir=fb.data_parameter_dir, result_dir=None, postfix=''):
    """
    Solve the single player game with the turn feature and tokens. Find the optimal policy to minimize the expected number of turns for reaching zero score. 
    Args: 
        name_pa: player ID
        data_parameter_dir=fb.data_parameter_dir          
        result_dir: folder to store the result 
        postfix='':
        gpu_device: None for CPU computation, otherwise use the gpu device ID defined in the system (default 0).
    Returns: 
        Either returns a dictionary or will save to the results directory if one is specified.
    """
    [aiming_grid, prob_grid_normalscore_nt, prob_grid_singlescore_nt, prob_grid_doublescore_nt, prob_grid_triplescore_nt, prob_grid_bullscore_nt] = load_aiming_grid(name_pa, epsilon=epsilon, data_parameter_dir=data_parameter_dir, grid_version='custom_no_tokens')
    [aiming_grid, prob_grid_normalscore_t, prob_grid_singlescore_t, prob_grid_doublescore_t, prob_grid_triplescore_t, prob_grid_bullscore_t] = load_aiming_grid('t', data_parameter_dir=data_parameter_dir, grid_version='custom_tokens')

    print('runing solve_dp_turn with credits')
    result_dic = solve_dp_turn_tokens(tokens, aiming_grid, prob_grid_normalscore_nt, prob_grid_singlescore_nt, prob_grid_doublescore_nt, prob_grid_triplescore_nt, prob_grid_bullscore_nt,prob_grid_normalscore_t, prob_grid_singlescore_t, prob_grid_doublescore_t, prob_grid_triplescore_t, prob_grid_bullscore_t)

    if (result_dir is not None):
        if not os.path.isdir(result_dir):
            os.makedirs(result_dir)        
        result_filename = result_dir + '/singlegame_results/singlegame_{}_e{}_turn_tokens{}.pkl'.format(name_pa, epsilon, postfix)
        ft.dump_pickle(result_filename, result_dic, printflag=True)
    else:
        return result_dic
