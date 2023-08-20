#%%
import os
import sys
import time

import numpy as np
import scipy.io as sio
from scipy.stats import multivariate_normal

# import os
# os.chdir("..")
# print(os.getcwd())

import function_init_board as fb
import function_tool as ft
import function_plot_board as ib
import function_init_simple_mdp as imdp


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

    result_dir = fb.data_parameter_dir + '/grid_full_augmented'
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)
    # player_parameter = sio.loadmat('./ALL_Model_Fits.mat')
    print(os.getcwd())
    player_parameter = sio.loadmat('./data_parameter/ALL_Model_Fits.mat')
    avg_model_t20 = sio.loadmat('./data_parameter/AVG_Model_Fits/ModelFit_T20_fixmu.mat')
    avg_model_t19 = sio.loadmat('./data_parameter/AVG_Model_Fits/ModelFit_T19_fixmu.mat')
    avg_model_t18 = sio.loadmat('./data_parameter/AVG_Model_Fits/ModelFit_T18_fixmu.mat')
    avg_model_t17 = sio.loadmat('./data_parameter/AVG_Model_Fits/ModelFit_T17_fixmu.mat')
    avg_model_b50 = sio.loadmat('./data_parameter/AVG_Model_Fits/ModelFit_B50_fixmu.mat')
    avg_model_AllDoubles = sio.loadmat('./data_parameter/AVG_Model_Fits/ModelFit_AllDoubles_fixmu.mat')
    
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

    singlescore_inner_grid = np.zeros((f_density_grid_num, f_density_grid_num), dtype=np.int8)
    singlescore_outer_grid = np.zeros((f_density_grid_num, f_density_grid_num), dtype=np.int8)

    for xi in f_density_xindex:
        for yi in f_density_yindex:
            singlescore_grid[yi,xi] = fb.get_score_singleonly(f_density_xgrid[xi], f_density_ygrid[yi])
            doublescore_grid[yi,xi] = fb.get_score_doubleonly(f_density_xgrid[xi], f_density_ygrid[yi])
            triplescore_grid[yi,xi] = fb.get_score_tripleonly(f_density_xgrid[xi], f_density_ygrid[yi])
            bullscore_grid[yi,xi] = fb.get_score_bullonly(f_density_xgrid[xi], f_density_ygrid[yi])

            singlescore_inner_grid[yi,xi] = fb.get_score_inner_singleonly(f_density_xgrid[xi], f_density_ygrid[yi])
            singlescore_outer_grid[yi,xi] = fb.get_score_outer_singleonly(f_density_xgrid[xi], f_density_ygrid[yi])

    singlescore_coordinate_dic = {}
    doublescore_coordinate_dic = {}
    triplescore_coordinate_dic = {}
    bullscore_coordinate_dic = {}
    
    singlescore_inner_coordinate_dic = {}
    singlescore_outer_coordinate_dic = {}

    ## coordinate for each score
    for si in range(20):
        singlescore_coordinate_dic[si] = np.where(singlescore_grid==fb.singlescorelist[si])
        doublescore_coordinate_dic[si] = np.where(doublescore_grid==fb.doublescorelist[si])
        triplescore_coordinate_dic[si] = np.where(triplescore_grid==fb.triplescorelist[si])

        singlescore_inner_coordinate_dic[si] = np.where(singlescore_inner_grid==fb.singlescorelist[si])
        singlescore_outer_coordinate_dic[si] = np.where(singlescore_outer_grid==fb.singlescorelist[si])

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

        prob_grid_singlescore_inner = np.zeros((grid_num, grid_num, fb.singlescorelist_len))
        prob_grid_singlescore_outer = np.zeros((grid_num, grid_num, fb.singlescorelist_len))
        
        if playerID == 'AVG':

            #### conduct a numerical integration to evaluate the hitting probability for each score associated with the given aiming location
            time1 = time.time()
            for xi in xindex:
                ##print(xi)
                for yi in yindex:
                    ## select the proper Gaussian distribution according to the area to which the aiming location belongs
                    mu = [xgrid[xi], ygrid[yi]]
                    score, multiplier = fb.get_score_and_multiplier(mu)
                    if (score==60 and multiplier==3): ##triple 20
                        covariance_matrix = avg_model_t20['ModelFit_T20_fixmu'][0][0][2] * epsilon
                    elif (score==57 and multiplier==3): ##triple 19
                        covariance_matrix = avg_model_t19['ModelFit_T19_fixmu'][0][0][2] * epsilon
                    elif (score==54 and multiplier==3): ##triple 18
                        ccovariance_matrix = avg_model_t18['ModelFit_T18_fixmu'][0][0][2] * epsilon 
                    elif (score==51 and multiplier==3): ##triple 17
                        covariance_matrix = avg_model_t17['ModelFit_T17_fixmu'][0][0][2] * epsilon 
                    elif (score==50 and multiplier==2): ##double bull
                        covariance_matrix = avg_model_b50['ModelFit_B50_fixmu'][0][0][2] * epsilon 
                    else:
                        covariance_matrix = avg_model_AllDoubles['ModelFit_AllDoubles_fixmu'][0][0][2] * epsilon 
                            
                    ## f_density_grid is the PDF of the fitted Gaussian distribution
                    rv = multivariate_normal(mu, covariance_matrix)
                    f_density_grid = rv.pdf(pos)
                
                    ## check score and integrate density
                    for si in range(20):
                        prob_grid_singlescore[xi,yi,si] = f_density_grid[singlescore_coordinate_dic[si]].sum()*f_density_constant
                        prob_grid_doublescore[xi,yi,si] = f_density_grid[doublescore_coordinate_dic[si]].sum()*f_density_constant
                        prob_grid_triplescore[xi,yi,si] = f_density_grid[triplescore_coordinate_dic[si]].sum()*f_density_constant

                        prob_grid_singlescore_inner[xi,yi,si] = f_density_grid[singlescore_inner_coordinate_dic[si]].sum()*f_density_constant
                        prob_grid_singlescore_outer[xi,yi,si] = f_density_grid[singlescore_outer_coordinate_dic[si]].sum()*f_density_constant


                    prob_grid_bullscore[xi,yi,0] = f_density_grid[bullscore_coordinate_dic[0]].sum()*f_density_constant
                    prob_grid_bullscore[xi,yi,1] = f_density_grid[bullscore_coordinate_dic[1]].sum()*f_density_constant
        
        else: 

            player_index = playerID - 1

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

                        prob_grid_singlescore_inner[xi,yi,si] = f_density_grid[singlescore_inner_coordinate_dic[si]].sum()*f_density_constant
                        prob_grid_singlescore_outer[xi,yi,si] = f_density_grid[singlescore_outer_coordinate_dic[si]].sum()*f_density_constant


                    prob_grid_bullscore[xi,yi,0] = f_density_grid[bullscore_coordinate_dic[0]].sum()*f_density_constant
                    prob_grid_bullscore[xi,yi,1] = f_density_grid[bullscore_coordinate_dic[1]].sum()*f_density_constant
                    
                
        result_dic = {'prob_grid_singlescore':prob_grid_singlescore, 'prob_grid_doublescore':prob_grid_doublescore,'prob_grid_triplescore':prob_grid_triplescore, 'prob_grid_bullscore':prob_grid_bullscore,'prob_grid_singlescore_inner':prob_grid_singlescore_inner,'prob_grid_singlescore_outer':prob_grid_singlescore_outer}    
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

    prob_grid_singlescore_inner = result_dic['prob_grid_singlescore_inner']
    prob_grid_singlescore_outer = result_dic['prob_grid_singlescore_outer']
    
    ## default setting counts bull score
    if count_bull:
        prob_grid_normalscore[:,fb.score_SB] += prob_grid_bullscore[:,0]
        prob_grid_normalscore[:,fb.score_DB] += prob_grid_bullscore[:,1]
    else:
        print('bull score in NOT counted in prob_grid_normalscore')
        
    return [aiming_grid, prob_grid_normalscore, prob_grid_singlescore, prob_grid_doublescore, prob_grid_triplescore, prob_grid_bullscore, prob_grid_singlescore_inner,prob_grid_singlescore_outer]


#%%
## 3-dimension probability grid
def load_prob_grid(playername_filename, epsilon=1, data_parameter_dir=fb.data_parameter_dir, grid_version='full_augmented'):
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
    prob_grid_singlescore_inner = prob_grid_dict['prob_grid_singlescore_inner']
    prob_grid_singlescore_outer = prob_grid_dict['prob_grid_singlescore_outer']
    
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
    
    return [prob_grid_normalscore, prob_grid_singlescore, prob_grid_doublescore, prob_grid_triplescore, prob_grid_bullscore,prob_grid_singlescore_inner,prob_grid_singlescore_outer]


#%%
def get_aiming_grid(playername_filename, epsilon=1, data_parameter_dir=fb.data_parameter_dir, grid_version='full_augmented'):

    temp_num = len(imdp.a_throw_list) + len(imdp.a_token_list)
    
    [prob_grid_normalscore, prob_grid_singlescore, prob_grid_doublescore, prob_grid_triplescore, prob_grid_bullscore,prob_grid_singlescore_inner,prob_grid_singlescore_outer] = load_prob_grid(playername_filename, epsilon=epsilon, data_parameter_dir=data_parameter_dir, grid_version=grid_version)    

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

    prob_grid_singlescore_inner_new = np.zeros((aiming_grid_num, 20))
    prob_grid_singlescore_outer_new = np.zeros((aiming_grid_num, 20))
    
    # regular actions 
    for temp_index in range(aiming_grid_throws):
        prob_grid_normalscore_new[temp_index,:] = prob_grid_normalscore[aiming_grid[temp_index,0], aiming_grid[temp_index,1], :]
        prob_grid_singlescore_new[temp_index,:] = prob_grid_singlescore[aiming_grid[temp_index,0], aiming_grid[temp_index,1], :]
        prob_grid_doublescore_new[temp_index,:] = prob_grid_doublescore[aiming_grid[temp_index,0], aiming_grid[temp_index,1], :]
        prob_grid_triplescore_new[temp_index,:] = prob_grid_triplescore[aiming_grid[temp_index,0], aiming_grid[temp_index,1], :]
        prob_grid_bullscore_new[temp_index,:] = prob_grid_bullscore[aiming_grid[temp_index,0], aiming_grid[temp_index,1], :]
        prob_grid_singlescore_inner_new[temp_index,:] = prob_grid_singlescore_inner[aiming_grid[temp_index,0], aiming_grid[temp_index,1], :]
        prob_grid_singlescore_outer_new[temp_index,:] = prob_grid_singlescore_outer[aiming_grid[temp_index,0], aiming_grid[temp_index,1], :]

    # token actions
    for temp_index in range(aiming_grid_throws,aiming_grid_num):
        prob_grid_normalscore_new[temp_index,0] = 1

    return [aiming_grid, prob_grid_normalscore_new, prob_grid_singlescore_new, prob_grid_doublescore_new, prob_grid_triplescore_new, prob_grid_bullscore_new, prob_grid_singlescore_inner_new, prob_grid_singlescore_outer_new]

#%%
## save custom action set with no tokens
def save_aiming_grid(playerID_list, epsilon=1,grid_version='full_augmented'):    
    grid_version_result = 'custom_augmented'
    print('generate and save action set grid_version={}'.format(grid_version_result))
    for playerID in playerID_list:
        name_pa = 'player{}'.format(playerID)    
        [aiming_grid, prob_grid_normalscore, prob_grid_singlescore, prob_grid_doublescore, prob_grid_triplescore, prob_grid_bullscore, prob_grid_singlescore_inner, prob_grid_singlescore_outer] = get_aiming_grid(name_pa, epsilon=epsilon, data_parameter_dir=fb.data_parameter_dir, grid_version=grid_version)
        
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

        result_dic['prob_grid_singlescore_inner'] = prob_grid_singlescore_inner
        result_dic['prob_grid_singlescore_outer'] = prob_grid_singlescore_outer
        
        result_dir = fb.data_parameter_dir + '/grid_{}'.format(grid_version_result)
        if not os.path.isdir(result_dir):
            os.makedirs(result_dir)        
        result_filename = result_dir + '/{}_e{}_gaussin_prob_grid_{}.pkl'.format(name_pa, epsilon, grid_version_result)
        ft.dump_pickle(result_filename, result_dic, printflag=True)
    
    print()
    return

