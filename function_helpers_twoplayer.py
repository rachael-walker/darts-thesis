import numpy as np
import function_init_board as fb
import function_init_simple_mdp as imdp
import function_helpers_singleplayer as h



def zsg_policy_evaluation_tokens(value_pa, value_pb, tokens_pa, score_state_pa, score_state_pb, prob_turn_transit_pa, prob_turn_transit_pb):
    """
    Compute the game value (in terms of Player A's winning probability) for a specific turn (score_state_pa, score_state_pb) given Player A and B's policies. 
    Args: 
        value_pa, value_pb: game values for [s_a < score_state_pa, s_b < score_state_pb] are already solved. 
        score_state_pa, score_state_pb: scores for Player A and B at the beginning of this turn
        prob_turn_transit_pa, prob_turn_transit_pb: state transition probability associated with the given policies. 
    
    Returns: 
        value_state_pa: Player A's winning probability when A throws in this turn
        value_state_pb: Player A's winning probability when B throws in this turn
    """
    
    value_win_pa = 1 # A win
    value_win_pb = 0 # A lose

    score_max_pa = min(score_state_pa-2, 3*fb.maxhitscore)
    score_max_pb = min(score_state_pb-2, 3*fb.maxhitscore)

    prob_score_pa = prob_turn_transit_pa['score']
    prob_score_pb = prob_turn_transit_pb['score']
    prob_finish_pa = prob_turn_transit_pa['finish']
    prob_finish_pb = prob_turn_transit_pb['finish']
    prob_zeroscore_pa = prob_turn_transit_pa['bust'][0] + prob_score_pa[0][0]
    prob_zeroscore_pb = prob_turn_transit_pb['bust'][0] + prob_score_pb[0][0]

    possible_tokens_used = prob_turn_transit_pa['bust'].shape[0] - 1

    # sa = sum(sum(prob_turn_transit_pa['score'])) + sum(prob_turn_transit_pa['bust']) + prob_turn_transit_pa['finish']
    # sb = sum(sum(prob_turn_transit_pb['score'])) + sum(prob_turn_transit_pb['bust']) + prob_turn_transit_pb['finish']
    # print(sa,sb)

    # Transit to end
    constant_pa = prob_finish_pa * value_win_pa

    # Only throws startying from Player A's turn, score greater than zero
    constant_pa += np.dot(prob_score_pa[0,1:], value_pb[tokens_pa, score_state_pa-1:score_state_pa-score_max_pa-1:-1, score_state_pb])        

    # Use at least one token:
    if tokens_pa > 0:

        # Go through subsequent scores after using at least one token 
        for i in range(0,possible_tokens_used):

            constant_pa += np.dot(np.flip(prob_score_pa[0:possible_tokens_used+1,:],axis=0)[i], value_pb[tokens_pa-possible_tokens_used:tokens_pa+1,score_state_pa:score_state_pa-score_max_pa-1:-1, score_state_pb][i])        

        # Go bust after using at least one token 
        constant_pa += np.dot(np.flip(prob_turn_transit_pa['bust'][1:possible_tokens_used+1]) , value_pb[tokens_pa-possible_tokens_used+1:tokens_pa+1,score_state_pa,score_state_pb])
        
    constant_pb = prob_finish_pb * value_win_pb #0
    constant_pb += np.dot(prob_score_pb[0][1:], value_pa[tokens_pa,score_state_pa, score_state_pb-1:score_state_pb-score_max_pb-1:-1])

    value_state_pa = (constant_pa+constant_pb*prob_zeroscore_pa)/(1-prob_zeroscore_pa*prob_zeroscore_pb)
    value_state_pb = constant_pb + value_state_pa*prob_zeroscore_pb
    
    return [value_state_pa, value_state_pb]



def load_ns_policy_dicts(name_pw,name_ps,epsilon_pw,epsilon_ps,dp_policy_folder,aiming_grid, prob_grid_normalscore_nt_pw, prob_grid_singlescore_nt_pw, prob_grid_doublescore_nt_pw, prob_grid_triplescore_nt_pw, prob_grid_bullscore_nt_pw, prob_grid_normalscore_nt_ps, prob_grid_singlescore_nt_ps, prob_grid_doublescore_nt_ps, prob_grid_triplescore_nt_ps, prob_grid_bullscore_nt_ps, prob_grid_normalscore_t, prob_grid_singlescore_t, prob_grid_doublescore_t, prob_grid_triplescore_t, prob_grid_bullscore_t):
     ## use single player game as the fixed policy    
    dp_policy_dict_pw = None
    dp_policy_dict_ps = None
    if dp_policy_folder is not None:
        dp_policy_filename_pw = dp_policy_folder + '/singlegame_results/singlegame_{}_e{}_turn_tokens.pkl'.format(name_pw,epsilon_pw)
        if (os.path.isfile(dp_policy_filename_pw) == True):
            dp_policy_dict_pw = ft.load_pickle(dp_policy_filename_pw)
            print('load weaker player policy {}'.format(dp_policy_filename_pw))
        dp_policy_filename_ps = dp_policy_folder + '/singlegame_results/singlegame_{}_e{}_turn_tokens.pkl'.format(name_ps,epsilon_ps)
        if (os.path.isfile(dp_policy_filename_ps) == True):
            dp_policy_dict_ps = ft.load_pickle(dp_policy_filename_ps)
            print('load stronger player policy {}'.format(dp_policy_filename_ps))    
    if dp_policy_dict_pw is None:
        print('solve weaker player NS policy')
        dp_policy_dict_pw = h.solve_dp_turn_tokens(9, aiming_grid, prob_grid_normalscore_nt_pw, prob_grid_singlescore_nt_pw, prob_grid_doublescore_nt_pw, prob_grid_triplescore_nt_pw, prob_grid_bullscore_nt_pw, prob_grid_normalscore_t, prob_grid_singlescore_t, prob_grid_doublescore_t, prob_grid_triplescore_t, prob_grid_bullscore_t)
    if dp_policy_dict_ps is None:
        print('solve stronger player NS policy')
        dp_policy_dict_ps = h.solve_dp_turn_tokens(9, aiming_grid, prob_grid_normalscore_nt_ps, prob_grid_singlescore_nt_ps, prob_grid_doublescore_nt_ps, prob_grid_triplescore_nt_ps, prob_grid_bullscore_nt_ps, prob_grid_normalscore_t, prob_grid_singlescore_t, prob_grid_doublescore_t, prob_grid_triplescore_t, prob_grid_bullscore_t)

    return dp_policy_dict_pw,dp_policy_dict_ps


def zsg_policy_improvement_tokens(param):
    """
    Do a policy improvement. Solve the Bellman equation. Find the best aiming location for each state in a turn using the last updated state values. 
    Args: 
        a dict param containing necessary informations. 
    
    Returns:     
        max_action_diff, max_value_relerror: relative errors after this policy iteration step 
    """
    #### input value ####
    prob_normalscore_tensor_nt = param['prob_normalscore_tensor_nt']
    prob_normalscore_tensor_t = param['prob_normalscore_tensor_t']
    prob_doublescore_dic_nt = param['prob_doublescore_dic_nt']
    prob_doublescore_dic_t = param['prob_doublescore_dic_t']
    prob_DB_nt = param['prob_DB_nt']
    prob_DB_t = param['prob_DB_t']
    prob_bust_dic_nt = param['prob_bust_dic_nt']
    prob_bust_dic_t = param['prob_bust_dic_t']
    num_actions = prob_normalscore_tensor_nt.shape[0]

    state_len_vector = param['state_len_vector']
    score_state = param['score_state']   
    token_state = param['token_state']  
    state_action = param['state_action']
    state_value = param['state_value']         
    state_action_update = param['state_action_update']
    state_value_update = param['state_value_update']
    action_diff = param['action_diff']
    value_relerror = param['value_relerror']

    flag_max = param['flag_max'] ## maximize or minimize 
    next_turn_value = param['next_turn_value']
    game_end_value = param['game_end_value']    
    if 'round_index' in param:
        round_index = param['round_index']
    else:
        round_index = 0

    #### policy improvement ####
    for rt in [1,2,3]:

        this_throw_state_len = state_len_vector[rt]
        state_notbust_len =  max(min(score_state-61, this_throw_state_len),0)
        token_index = min(2,token_state+1)

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

                    new_state_vals_nt = next_turn_value[token_state]
                    next_state_value_array_nt[:,score_gained] = new_state_vals_nt[score_remain:score_remain-score_max_plus1:-1]

                    if token_state > 0: 
                        new_state_vals_t = next_turn_value[token_state-1]
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
                    
                    new_state_vals_nt = state_value_update[rt-1][token_state]
                    next_state_value_array_nt[:,score_gained] = new_state_vals_nt[score_gained:score_gained+score_max_plus1]

                    if token_state > 0: 
                        new_state_vals_t = state_value_update[rt-1][token_state-1]
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

                    new_state_vals_nt = state_value_update[rt-1][token_state]
                    next_state_value_array_nt[:] = new_state_vals_nt[score_gained:score_gained+score_max_plus1]
                    
                    if token_state > 0:
                        new_state_vals_t = state_value_update[rt-1][token_state-1]
                        next_state_value_array_t[:] = new_state_vals_t[score_gained:score_gained+score_max_plus1]
                
                ## transit to next turn when rt=1
                else:

                    new_state_vals_nt = next_turn_value[token_state]
                    next_state_value_array_nt[:] = new_state_vals_nt[score_remain:score_remain-score_max_plus1:-1]
                    
                    if token_state > 0: 
                        new_state_vals_t = next_turn_value[token_state-1]
                        next_state_value_array_t[:] = new_state_vals_t[score_remain:score_remain-score_max_plus1:-1]
                        
            ## matrix product to compute all together
            next_state_value_tensor_nt = torch.from_numpy(next_state_value_array_nt)
            next_state_value_tensor_t = torch.from_numpy(next_state_value_array_t)

            try: 
                win_prob_tensor = torch.zeros((num_actions,next_state_value_array_nt.shape[1]))
            except:
                win_prob_tensor = torch.zeros(num_actions)
            
            # if we have tokens 
            if token_state > 0:  

                # use no token probabilities for no token actions 
                win_prob_tensor[:imdp.throw_num] += prob_normalscore_tensor_nt[:imdp.throw_num].matmul(next_state_value_tensor_nt)
                # use token probabilities for token actions 
                win_prob_tensor[imdp.throw_num:num_actions] += prob_normalscore_tensor_t[imdp.throw_num:].matmul(next_state_value_tensor_t)

            # if we don't have tokens   
            else:
                # add no token expectation --> index is 1 because we keep the same # of tokens       
                win_prob_tensor = prob_normalscore_tensor_nt.matmul(next_state_value_tensor_nt)

            ## searching
            if flag_max:
                if token_state == 0: 
                    temp1 = win_prob_tensor[:imdp.throw_num].max(axis=0)
                else: 
                    temp1 = win_prob_tensor.max(axis=0) 
            else:
                if token_state == 0: 
                    temp1 = win_prob_tensor[:imdp.throw_num].min(axis=0)
                else: 
                    temp1 = win_prob_tensor.min(axis=0) 

            state_action_update[rt][token_state][0:state_notbust_update_index] = temp1.indices.numpy()
            state_value_update[rt][token_state][0:state_notbust_update_index] =  temp1.values.numpy()                
        
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
                    
                    new_state_vals_nt = state_value_update[rt-1][token_state]
                    next_state_value_array_nt[0:score_max_plus1,score_gained_index] = new_state_vals_nt[score_gained:score_gained+score_max_plus1]

                    if token_state > 0: 
                        new_state_vals_t = state_value_update[rt-1][token_state-1]
                        next_state_value_array_t[0:score_max_plus1,score_gained_index] = new_state_vals_t[score_gained:score_gained+score_max_plus1]

                ## transit to next turn when rt=1
                else:

                    new_state_vals_nt = next_turn_value[token_state]
                    next_state_value_array_nt[0:score_max_plus1,score_gained_index] = new_state_vals_nt[score_remain:score_remain-score_max_plus1:-1]

                    
                    if token_state > 0: 
                        new_state_vals_t = next_turn_value[token_state-1]
                        next_state_value_array_t[0:score_max_plus1,score_gained_index] = new_state_vals_t[score_remain:score_remain-score_max_plus1:-1]
                        
            ## matrix product to compute all together
            next_state_value_tensor_nt = torch.from_numpy(next_state_value_array_nt)
            next_state_value_tensor_t = torch.from_numpy(next_state_value_array_t)

            # initialized
            try: 
                win_prob_tensor = torch.zeros((num_actions,next_state_value_array_nt.shape[1]))
            except:
                win_prob_tensor = torch.zeros(num_actions)
        
            # if we have tokens 
            if token_state > 0: 
                # use no token probabilities for no token actions
                win_prob_tensor[:imdp.throw_num] += prob_normalscore_tensor_nt[:imdp.throw_num].matmul(next_state_value_tensor_nt)
                
                # use token probabilities for token actions 
                win_prob_tensor[imdp.throw_num:] += prob_normalscore_tensor_t[imdp.throw_num:].matmul(next_state_value_tensor_t)

            else: 
                # add no token expectation  
                win_prob_tensor += prob_normalscore_tensor_nt.matmul(next_state_value_tensor_nt)

            ## consider bust/finishing for each bust state seperately 
            win_prob_array = win_prob_tensor.numpy()                
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
                if (score_remain == fb.score_DB):                        
                    win_prob_array[:imdp.throw_num,score_gained_index] += prob_DB_nt[:imdp.throw_num]*game_end_value
                    win_prob_array[imdp.throw_num:,score_gained_index] += prob_DB_t[imdp.throw_num:]*game_end_value
                elif (score_remain <= 40 and score_remain%2==0):
                    win_prob_array[:imdp.throw_num,score_gained_index] += prob_doublescore_dic_nt[score_remain][:imdp.throw_num]*game_end_value
                    win_prob_array[imdp.throw_num:,score_gained_index] += prob_doublescore_dic_t[score_remain][imdp.throw_num:]*game_end_value
                else:
                    pass  

                ## transit to bust
                win_prob_array[:imdp.throw_num,score_gained_index] += prob_bust_dic_nt[score_max][:imdp.throw_num]*(next_turn_value[token_state][score_state])  ## 1 turn is already counted before
                win_prob_array[imdp.throw_num:,score_gained_index] += prob_bust_dic_t[score_max][imdp.throw_num:]*(next_turn_value[token_state-1][score_state])
                    
            ## searching
            if flag_max:
                if token_state == 0: 
                    temp1 = win_prob_tensor[:imdp.throw_num].max(axis=0)
                else: 
                    temp1 = win_prob_tensor.max(axis=0)
            else:
                if token_state == 0: 
                    temp1 = win_prob_tensor[:imdp.throw_num].min(axis=0)
                else: 
                    temp1 = win_prob_tensor.min(axis=0)
            state_action_update[rt][token_state][state_notbust_len:this_throw_state_len] = temp1.indices.numpy()
            state_value_update[rt][token_state][state_notbust_len:this_throw_state_len] =  temp1.values.numpy()                

        #### finish rt=1,2,3. check improvement
        action_diff[rt][token_state][:] = np.abs(state_action_update[rt][token_state] - state_action[rt][token_state])                                
        value_relerror[rt] = np.abs((state_value_update[rt] - state_value[rt])/state_value_update[rt]).max()
        state_action[rt][token_state][:] = state_action_update[rt][token_state][:]
        state_value[rt][token_state][:] = state_value_update[rt][token_state][:]

    max_action_diff = max([action_diff[1].max(), action_diff[2].max(), action_diff[3].max()])
    max_value_relerror = value_relerror.max()

    return [max_action_diff, max_value_relerror]


def solve_zsg_optW_fixNS(name_pw, name_ps, epsilon_pw, epsilon_ps, max_tokens_optimize=9, data_parameter_dir=fb.data_parameter_dir, dp_policy_folder=None, result_dir=None, postfix='', gpu_device=None):
    
    max_tokens=max_tokens_optimize
    game_begin_score_502 = 501+1
    
    info = 'W_{}e{}_S_{}e{}_optW'.format(name_pw, epsilon_pw, name_ps, epsilon_ps)
    print(info)
    ##
    if result_dir is not None:
        if not os.path.isdir(result_dir):
            os.makedirs(result_dir)
        result_filename = result_dir + '/zsg_W_{}e{}_S_{}e{}_{}_optW.pkl'.format(name_pw, epsilon_pw, name_ps, epsilon_ps, postfix)
        result_value_filename = result_dir + '/zsg_value_W_{}e{}_S_{}e{}_{}_optW.pkl'.format(name_pw, epsilon_pw, name_ps, epsilon_ps, postfix)


    [aiming_grid, prob_grid_normalscore_nt_pw, prob_grid_singlescore_nt_pw, prob_grid_doublescore_nt_pw, prob_grid_triplescore_nt_pw, prob_grid_bullscore_nt_pw] = h.load_aiming_grid(name_pw, epsilon=epsilon_pw, data_parameter_dir=data_parameter_dir, grid_version='custom_no_tokens')
    [aiming_grid, prob_grid_normalscore_nt_ps, prob_grid_singlescore_nt_ps, prob_grid_doublescore_nt_ps, prob_grid_triplescore_nt_ps, prob_grid_bullscore_nt_ps] = h.load_aiming_grid(name_ps, epsilon=epsilon_ps, data_parameter_dir=data_parameter_dir, grid_version='custom_no_tokens')
    [aiming_grid, prob_grid_normalscore_t, prob_grid_singlescore_t, prob_grid_doublescore_t, prob_grid_triplescore_t, prob_grid_bullscore_t] = h.load_aiming_grid('t', data_parameter_dir=data_parameter_dir, grid_version='custom_tokens')

    ## use single player game as the fixed policy    
    dp_policy_dict_pw, dp_policy_dict_ps = load_ns_policy_dicts(name_pw,name_ps,epsilon_pw,epsilon_ps,dp_policy_folder,aiming_grid, prob_grid_normalscore_nt_pw, prob_grid_singlescore_nt_pw, prob_grid_doublescore_nt_pw, prob_grid_triplescore_nt_pw, prob_grid_bullscore_nt_pw, prob_grid_normalscore_nt_ps, prob_grid_singlescore_nt_ps, prob_grid_doublescore_nt_ps, prob_grid_triplescore_nt_ps, prob_grid_bullscore_nt_ps, prob_grid_normalscore_t, prob_grid_singlescore_t, prob_grid_doublescore_t, prob_grid_triplescore_t, prob_grid_bullscore_t)


    num_aiming_location_pw, prob_normalscore_nt_pw, prob_doublescore_dic_nt_pw, prob_DB_nt_pw, prob_bust_dic_nt_pw, prob_notbust_dic_nt_pw, prob_normalscore_t, prob_doublescore_dic_t, prob_DB_t, prob_bust_dic_t, prob_notbust_dic_t = h.init_probabilities(aiming_grid, prob_grid_normalscore_nt_pw, prob_grid_doublescore_nt_pw, prob_grid_bullscore_nt_pw, prob_grid_normalscore_t, prob_grid_doublescore_t, prob_grid_bullscore_t)

    param_pw = {}    

    prob_normalscore_tensor_nt_pw = torch.from_numpy(prob_normalscore_nt_pw)
    prob_normalscore_tensor_t = torch.from_numpy(prob_normalscore_t)
    param_pw['prob_normalscore_tensor_nt'] = prob_normalscore_tensor_nt_pw
    param_pw['prob_normalscore_tensor_t'] = prob_normalscore_tensor_t
    param_pw['prob_doublescore_dic_nt'] = prob_doublescore_dic_nt_pw
    param_pw['prob_doublescore_dic_t'] = prob_doublescore_dic_t
    param_pw['prob_DB_nt'] = prob_DB_nt_pw
    param_pw['prob_DB_t'] = prob_DB_t
    param_pw['prob_bust_dic_nt'] = prob_bust_dic_nt_pw
    param_pw['prob_bust_dic_t'] = prob_bust_dic_t
        
    #### 
    iteration_round_limit = 20
    iteration_relerror_limit = 10**-9

    value_pw = np.zeros((max_tokens+1,game_begin_score_502,game_begin_score_502))  # player A's winning probability when A throws at state [score_A, score_B]
    value_ps = np.zeros((max_tokens+1,game_begin_score_502,game_begin_score_502))  # player A's winning probability when B throws at state [score_A, score_B]
    value_win_pw = 1.0
    value_win_ps = 0.0
    num_iteration_record_pw = np.zeros((max_tokens+1,game_begin_score_502,game_begin_score_502), dtype=np.int8)

    state_len_vector_pw = np.zeros(4, dtype=np.int32)
    state_value_default  = [None]  ## expected # of turns for each state in the turn
    action_diff_pw  = [None]
    value_relerror_pw = np.zeros(4)

    for rt in [1,2,3]:
        ## for rt=3: possible score_gained = 0
        ## for rt=2: possible score_gained = 0,1,...,60
        ## for rt=1: possible score_gained = 0,1,...,120
        this_throw_state_len = fb.maxhitscore*(3-rt) + 1
        state_value_default.append(np.ones((max_tokens+1,this_throw_state_len))*fb.largenumber)
        action_diff_pw.append(np.ones((max_tokens+1,this_throw_state_len)))

    ## for player A. (player B is fixed)
    ## first key: score_state_pa=2,...,501; second key: score_state_pb=2,...,501; thrid key: throws=3,2,1
    optimal_value_dic = {} 
    optimal_action_index_dic = {}
    prob_turn_transit_dic_pw = {}
    for score in range(2,502):
        optimal_value_dic[score] = {}
        optimal_action_index_dic[score] = {}
        prob_turn_transit_dic_pw[score] = {}

    #### algorithm start ####
    t_policy_improvement = 0
    t_policy_evaluation = 0
    t_other = 0
    t1 = time.time()
    for score_state_ps in range(2, game_begin_score_502):
        t_scoreloop_begin = time.time()
        print('stronger state:',score_state_ps,'time:',t_scoreloop_begin-t1)
        score_state_list = []

        ## fix player B score, loop through player A
        for score_state_pw in range(2, game_begin_score_502):

            for tokens_pw in range(0,max_tokens+1):
                
                score_state_list.append([tokens_pw, score_state_pw, score_state_ps])

        for [tokens_pw, score_state_pw, score_state_ps] in score_state_list:
            # print('state',tokens_pw,score_state_pw,score_state_ps)

            ## initialize player A initial policy:
            for rt in [1,2,3]:        
                this_throw_state_len_pw = min(score_state_pw-2, fb.maxhitscore*(3-rt)) + 1
                state_len_vector_pw[rt] = this_throw_state_len_pw
            state_value_pw = ft.copy_numberarray_container(state_value_default)
            if score_state_ps > 2:
                state_action_pw = ft.copy_numberarray_container(optimal_action_index_dic[score_state_pw][score_state_ps-1])
                prob_turn_transit_pw = prob_turn_transit_dic_pw[score_state_pw][score_state_ps-1]
            else:
                state_action_pw = ft.copy_numberarray_container(dp_policy_dict_pw['optimal_action_index_dic'][score_state_pw])
                prob_turn_transit_pw = dp_policy_dict_pw['prob_scorestate_transit'][tokens_pw][score_state_pw]
            state_value_update_pw = ft.copy_numberarray_container(state_value_pw)
            state_action_update_pw = ft.copy_numberarray_container(state_action_pw)

            ## player B, turn score transit probability is fixed
            prob_turn_transit_ps = dp_policy_dict_ps['prob_scorestate_transit'][0][score_state_ps] # assume no tokens 

            ## assemble variables
            ## player A
            param_pw['state_len_vector'] = state_len_vector_pw
            param_pw['score_state'] = score_state_pw  
            param_pw['token_state'] = tokens_pw   
            param_pw['state_action'] = state_action_pw
            param_pw['state_value'] = state_value_pw
            param_pw['state_action_update'] = state_action_update_pw
            param_pw['state_value_update'] = state_value_update_pw   
            param_pw['action_diff'] = action_diff_pw
            param_pw['value_relerror'] = value_relerror_pw        
            ## maximize player A's win_prob
            param_pw['flag_max'] = True
            param_pw['next_turn_value'] = value_ps[:,score_state_ps] ## player B throws in next turn
            param_pw['game_end_value'] = value_win_pw

            ## policy iteration
            for round_index in range(iteration_round_limit):            
                
                #### policy evaluation ####
                tpe1 = time.time()
                ## evaluate current policy, player A winning probability at (score_pa, score_pb, i=3, u=0)
                ## value_pa: player A throws first, value_pb: player A throws second 
                ## player A, turn score transit probability                
                ## use the initial prob_turn_transit_pa value for round_index=0
                if (round_index >=0):
                    prob_turn_transit_pw = h.solve_turn_transit_probability_fast_token(score_state=score_state_pw,state_action=state_action_pw,available_tokens=tokens_pw,prob_grid_normalscore_nt=prob_grid_normalscore_nt_pw,prob_grid_doublescore_nt=prob_grid_doublescore_nt_pw,prob_grid_bullscore_nt=prob_grid_bullscore_nt_pw,prob_bust_dic_nt=prob_bust_dic_nt_pw,prob_grid_normalscore_t=prob_grid_normalscore_t,prob_grid_doublescore_t=prob_grid_doublescore_t,prob_grid_bullscore_t=prob_grid_bullscore_t,prob_bust_dic_t=prob_bust_dic_t)
                [value_state_pw, value_state_ps] = zsg_policy_evaluation_tokens(value_pw, value_ps, tokens_pw, score_state_pw, score_state_ps, prob_turn_transit_pw, prob_turn_transit_ps)
                value_pw[tokens_pw, score_state_pw, score_state_ps] = value_state_pw
                value_ps[tokens_pw, score_state_pw, score_state_ps] = value_state_ps
                tpe2 = time.time()
                t_policy_evaluation += (tpe2-tpe1) 

                #### policy improvement for player A ####
                tpi1 = time.time()
                param_pw['round_index'] = round_index
                [max_action_diff, max_value_relerror] = zsg_policy_improvement_tokens(param_pw)
                tpi2 = time.time()
                t_policy_improvement += (tpi2 - tpi1)                
                if (max_action_diff < 1):
                    break
                if (max_value_relerror < iteration_relerror_limit):
                    break

            optimal_action_index_dic[score_state_pw][score_state_ps] = state_action_pw
            optimal_value_dic[score_state_pw][score_state_ps] = state_value_pw
            prob_turn_transit_dic_pw[score_state_pw][score_state_ps] = prob_turn_transit_pw
            num_iteration_record_pw[tokens_pw, score_state_pw, score_state_ps] = round_index + 1

    ## computation is done
    t2 = time.time()
    print('solve_zsg_opt_{}e{}_fix_{}e{} in {} seconds'.format(name_pw,epsilon_pw, name_ps, epsilon_ps, t2-t1))
    print('t_policy_evaluation  = {} seconds'.format(t_policy_evaluation))
    print('t_policy_improvement = {} seconds'.format(t_policy_improvement))
    print('t_other = {} seconds'.format(t_other))    
    #print('value_pa {} '.format(value_pa))
    #print('value_pb {} '.format(value_pb))

    result_dic = {'optimal_action_index_dic':optimal_action_index_dic, 'value_pw':value_pw, 'value_ps':value_ps,'optimal_value_dic':optimal_value_dic, 'info':info}    
    if (result_dir is not None):
        ft.dump_pickle(result_filename, result_dic)
        print('save {}'.format(result_filename))
        ft.dump_pickle(result_value_filename, {'value_pw':value_pw, 'value_ps':value_ps, 'info':info})
        print('save {}'.format(result_value_filename))
        return 'save'
    else:
        return result_dic


## fix player A's Naive Strategy (NS) and optimize player B
def solve_zsg_optS_fixNS(name_pw, name_ps, epsilon_pw, epsilon_ps, data_parameter_dir=fb.data_parameter_dir, dp_policy_folder=None, result_dir=None, postfix='', gpu_device=None):
    
    max_tokens_stronger=0
    
    info = 'W_{}e{}_S_{}e{}_optS'.format(name_pw, epsilon_pw, name_ps, epsilon_ps)
    print(info)
    ##
    if result_dir is not None:    
        if not os.path.isdir(result_dir):
            os.makedirs(result_dir)
        result_filename = result_dir + '/zsg_W_{}e{}_S_{}e{}_{}_optS.pkl'.format(name_pw, epsilon_pw, name_ps, epsilon_ps, postfix)
        result_value_filename = result_dir + '/zsg_value_W_{}e{}_S_{}e{}_{}_optS.pkl'.format(name_pw, epsilon_pw, name_ps, epsilon_ps, postfix)

    ## need to reset the result key name: Player A is name_pb and Player B is name_pa
    ## game values are represented in terms of Player A's winning probability
    temp_result_dic = solve_zsg_optW_fixNS(name_ps, name_pw, epsilon_ps, epsilon_pw, max_tokens_optimize=max_tokens_stronger, dp_policy_folder=dp_policy_folder, result_dir=None, postfix='', gpu_device=gpu_device)
    result_dic = {'info':info}
    value_pw = 1-temp_result_dic['value_ps'].T    
    value_ps = 1-temp_result_dic['value_pw'].T
    value_pw[:2,:] = 0
    value_pw[:,:2] = 0
    value_pw[:2,:] = 0
    value_pw[:,:2] = 0
    
    result_dic['value_pw'] = value_pw
    result_dic['value_ps'] = value_ps   
    result_dic['optimal_action_index_dic'] = {}
    result_dic['optimal_value_dic'] = {}    
    for score_state_pw in range(2, game_begin_score_502):
    #for score_state_pa in range(2, 101):
        result_dic['optimal_action_index_dic'][score_state_pw] = {}
        result_dic['optimal_value_dic'][score_state_pw] = {}
        for score_state_ps in range(2, game_begin_score_502):
        #for score_state_pb in range(2, 101):
            result_dic['optimal_action_index_dic'][score_state_pw][score_state_ps] = temp_result_dic['optimal_action_index_dic'][score_state_ps][score_state_pw]
            result_dic['optimal_value_dic'][score_state_pw][score_state_ps] = temp_result_dic['optimal_value_dic'][score_state_ps][score_state_pw]
    
    if (result_dir is not None):
        ft.dump_pickle(result_filename, result_dic)
        print('save {}'.format(result_filename))
        ft.dump_pickle(result_value_filename, {'value_pa':value_pw, 'value_pb':value_ps, 'info':info})
        print('save {}'.format(result_value_filename))
        return 'save'
    else:
        return result_dic
    

## optimize player A and B alternatively until achieving optimal
def solve_zsg_optboth(name_pw, name_ps, epsilon_pw, epsilon_ps, data_parameter_dir=fb.data_parameter_dir, dp_policy_folder=None, result_dir=None, postfix='', gpu_device=None):
    info = 'W_{}e{}_S_{}e{}_optboth'.format(name_pw, epsilon_pw, name_ps, epsilon_ps)
    print(info)
    ##
    if result_dir is not None:
        if not os.path.isdir(result_dir):
            os.makedirs(result_dir)
        result_filename = result_dir + '/zsg_W_{}e{}_S_{}e{}_{}_optboth.pkl'.format(name_pw, epsilon_pw, name_ps, epsilon_ps, postfix)
        result_value_filename = result_dir + '/zsg_value_W_{}e{}_S_{}e{}_{}_optboth.pkl'.format(name_pw, epsilon_pw, name_ps, epsilon_ps, postfix)


    max_tokens = 9
    game_begin_score_502 = 501+1
    #player A: pa throw first
    #player B: pb throw after player A, policy is fixed as ns
    print('player W is {} e{} and player S is {} e{}'.format(name_pw, epsilon_pw, name_ps, epsilon_ps))
    print('optimize both players')

    [aiming_grid, prob_grid_normalscore_nt_pw, prob_grid_singlescore_nt_pw, prob_grid_doublescore_nt_pw, prob_grid_triplescore_nt_pw, prob_grid_bullscore_nt_pw] = h.load_aiming_grid(name_pw, epsilon=epsilon_pw, data_parameter_dir=data_parameter_dir, grid_version='custom_no_tokens')
    [aiming_grid, prob_grid_normalscore_nt_ps, prob_grid_singlescore_nt_ps, prob_grid_doublescore_nt_ps, prob_grid_triplescore_nt_ps, prob_grid_bullscore_nt_ps] = h.load_aiming_grid(name_ps, epsilon=epsilon_ps, data_parameter_dir=data_parameter_dir, grid_version='custom_no_tokens')
    [aiming_grid, prob_grid_normalscore_t, prob_grid_singlescore_t, prob_grid_doublescore_t, prob_grid_triplescore_t, prob_grid_bullscore_t] = h.load_aiming_grid('t', data_parameter_dir=data_parameter_dir, grid_version='custom_tokens')

    ## use single player game as the fixed policy    
    dp_policy_dict_pw, dp_policy_dict_ps = load_ns_policy_dicts(name_pw,name_ps,epsilon_pw,epsilon_ps,dp_policy_folder,aiming_grid, prob_grid_normalscore_nt_pw, prob_grid_singlescore_nt_pw, prob_grid_doublescore_nt_pw, prob_grid_triplescore_nt_pw, prob_grid_bullscore_nt_pw, prob_grid_normalscore_nt_ps, prob_grid_singlescore_nt_ps, prob_grid_doublescore_nt_ps, prob_grid_triplescore_nt_ps, prob_grid_bullscore_nt_ps, prob_grid_normalscore_t, prob_grid_singlescore_t, prob_grid_doublescore_t, prob_grid_triplescore_t, prob_grid_bullscore_t)

    #### data for player A ####
    num_aiming_location_pw, prob_normalscore_nt_pw, prob_doublescore_dic_nt_pw, prob_DB_nt_pw, prob_bust_dic_nt_pw, prob_notbust_dic_nt_pw, prob_normalscore_t, prob_doublescore_dic_t, prob_DB_t, prob_bust_dic_t, prob_notbust_dic_t = h.init_probabilities(aiming_grid, prob_grid_normalscore_nt_pw, prob_grid_doublescore_nt_pw, prob_grid_bullscore_nt_pw, prob_grid_normalscore_t, prob_grid_doublescore_t, prob_grid_bullscore_t)

    param_pw = {}    

    prob_normalscore_tensor_nt_pw = torch.from_numpy(prob_normalscore_nt_pw)
    prob_normalscore_tensor_t = torch.from_numpy(prob_normalscore_t)
    param_pw['prob_normalscore_tensor_nt'] = prob_normalscore_tensor_nt_pw
    param_pw['prob_normalscore_tensor_t'] = prob_normalscore_tensor_t
    param_pw['prob_doublescore_dic_nt'] = prob_doublescore_dic_nt_pw
    param_pw['prob_doublescore_dic_t'] = prob_doublescore_dic_t
    param_pw['prob_DB_nt'] = prob_DB_nt_pw
    param_pw['prob_DB_t'] = prob_DB_t
    param_pw['prob_bust_dic_nt'] = prob_bust_dic_nt_pw
    param_pw['prob_bust_dic_t'] = prob_bust_dic_t

    #### data for player B ####
    num_aiming_location_ps, prob_normalscore_nt_ps, prob_doublescore_dic_nt_ps, prob_DB_nt_ps, prob_bust_dic_nt_ps, prob_notbust_dic_nt_ps, prob_normalscore_t, prob_doublescore_dic_t, prob_DB_t, prob_bust_dic_t, prob_notbust_dic_t = h.init_probabilities(aiming_grid, prob_grid_normalscore_nt_ps, prob_grid_doublescore_nt_ps, prob_grid_bullscore_nt_ps, prob_grid_normalscore_t, prob_grid_doublescore_t, prob_grid_bullscore_t)

    param_ps = {}    

    prob_normalscore_tensor_nt_ps = torch.from_numpy(prob_normalscore_nt_ps)
    prob_normalscore_tensor_t = torch.from_numpy(prob_normalscore_t)
    param_ps['prob_normalscore_tensor_nt'] = prob_normalscore_tensor_nt_ps
    param_ps['prob_normalscore_tensor_t'] = prob_normalscore_tensor_t
    param_ps['prob_doublescore_dic_nt'] = prob_doublescore_dic_nt_ps
    param_ps['prob_doublescore_dic_t'] = prob_doublescore_dic_t
    param_ps['prob_DB_nt'] = prob_DB_nt_ps
    param_ps['prob_DB_t'] = prob_DB_t
    param_ps['prob_bust_dic_nt'] = prob_bust_dic_nt_ps
    param_ps['prob_bust_dic_t'] = prob_bust_dic_t
        
    #### 
    iteration_round_limit_zsgtwoplayers = 5
    iteration_relerror_limit_zsgtwoplayers = 10**-9
    iteration_round_zsgtwoplayers = np.zeros((max_tokens+1,502,502), dtype=np.int8)

    iteration_round_limit_singleplayer_policy = 20
    iteration_relerror_limit_singleplayer_policy = 10**-9

    value_pw = np.zeros((max_tokens+1,502,502))  # player A's winning probability when A throws at state [score_A, score_B]
    value_ps = np.zeros((max_tokens+1,502,502))  # player A's winning probability when B throws at state [score_A, score_B]
    value_win_pw = 1.0
    value_win_ps = 0.0
    num_iteration_record_pw = np.zeros((max_tokens+1,502,502), dtype=np.int8)
    num_iteration_record_ps = np.zeros((max_tokens+1,502,502), dtype=np.int8)
    ## values when optimizing A
    value_pw_optW = value_pw.copy()
    value_ps_optW = value_ps.copy()
    ## values when optimizing B
    value_pw_optS = value_pw.copy()
    value_ps_optS = value_ps.copy()    

    state_len_vector_pw = np.zeros(4, dtype=np.int32)
    state_value_default  = [None]  
    action_diff_pw  = [None]
    value_relerror_pw = np.zeros(4)
    for rt in [1,2,3]:
        ## for rt=3: possible score_gained = 0
        ## for rt=2: possible score_gained = 0,1,...,60
        ## for rt=1: possible score_gained = 0,1,...,120
        this_throw_state_len = fb.maxhitscore*(3-rt) + 1
        state_value_default.append(np.ones((max_tokens+1,this_throw_state_len))*fb.largenumber)
        action_diff_pw.append(np.ones((max_tokens+1,this_throw_state_len)))    
    state_len_vector_ps = np.zeros(4, dtype=np.int32)
    action_diff_ps = ft.copy_numberarray_container(action_diff_pw)
    value_relerror_ps = np.zeros(4)

    optimal_value_dic_pw = {}
    optimal_action_index_dic_pw = {}
    prob_turn_transit_dic_pw = {}
    optimal_value_dic_ps = {} 
    optimal_action_index_dic_ps = {}
    prob_turn_transit_dic_ps = {}

    for score in range(2,502):
        optimal_value_dic_pw[score] = {}
        optimal_action_index_dic_pw[score] = {}
        prob_turn_transit_dic_pw[score] = {}
        optimal_value_dic_ps[score] = {}
        optimal_action_index_dic_ps[score] = {}
        prob_turn_transit_dic_ps[score] = {}

    #### algorithm start ####
    t_policy_improvement = 0
    t_policy_evaluation = 0
    t_other = 0
    t1 = time.time()
    for score_state_ps in range(2, game_begin_score_502):
        t_scoreloop_begin = time.time()
        print('stronger state:',score_state_ps,'time:',t_scoreloop_begin-t1)
        score_state_list = []

        ## fix player B score, loop through player A
        for score_state_pw in range(2, game_begin_score_502):

            for tokens_pw in range(0,max_tokens+1):
                
                score_state_list.append([tokens_pw, score_state_pw, score_state_ps])

        ########     solve all states in turn [score_A, score_B]    ########
        for [tokens_pw, score_state_pw, score_state_ps] in score_state_list:
            #print('##### score_state [score_pa, score_pb] = {} ####'.format([score_state_pa, score_state_pb]))

            ## initialize the starting policy:
            ## player A
            for rt in [1,2,3]:        
                this_throw_state_len_pw = min(score_state_pw-2, fb.maxhitscore*(3-rt)) + 1
                state_len_vector_pw[rt] = this_throw_state_len_pw
            state_value_pw = ft.copy_numberarray_container(state_value_default)
            if score_state_ps > 2:
                state_action_pw = ft.copy_numberarray_container(optimal_action_index_dic_pw[score_state_pw][score_state_ps-1])
                prob_turn_transit_pw = prob_turn_transit_dic_pw[score_state_pw][score_state_ps-1]
            else:
                state_action_pw = ft.copy_numberarray_container(dp_policy_dict_pw['optimal_action_index_dic'][score_state_pw])
                prob_turn_transit_pw = dp_policy_dict_pw['prob_scorestate_transit'][tokens_pw][score_state_pw]
            state_value_update_pw = ft.copy_numberarray_container(state_value_pw)
            state_action_update_pw = ft.copy_numberarray_container(state_action_pw)


            ## player B
            for rt in [1,2,3]:        
                this_throw_state_len_ps = min(score_state_ps-2, fb.maxhitscore*(3-rt)) + 1
                state_len_vector_ps[rt] = this_throw_state_len_ps
            state_value_ps = ft.copy_numberarray_container(state_value_default)
            if score_state_pw > 2:
                state_action_ps = ft.copy_numberarray_container(optimal_action_index_dic_ps[score_state_pw-1][score_state_ps])
                prob_turn_transit_ps = prob_turn_transit_dic_ps[score_state_pw-1][score_state_ps]
            else:
                state_action_ps = ft.copy_numberarray_container(dp_policy_dict_ps['optimal_action_index_dic'][score_state_ps])
                prob_turn_transit_ps = dp_policy_dict_ps['prob_scorestate_transit'][0][score_state_ps]
            state_value_update_ps = ft.copy_numberarray_container(state_value_ps)
            state_action_update_ps = ft.copy_numberarray_container(state_action_ps)
            
            
            ## assemble variables
            ## player A
            param_pw['score_state'] = score_state_pw   
            param_pw['token_state'] = tokens_pw   
            param_pw['state_len_vector'] = state_len_vector_pw        
            param_pw['state_action'] = state_action_pw
            param_pw['state_value'] = state_value_pw
            param_pw['state_action_update'] = state_action_update_pw
            param_pw['state_value_update'] = state_value_update_pw
            param_pw['action_diff'] = action_diff_pw
            param_pw['value_relerror'] = value_relerror_pw   
            ## maximize player A's win_prob
            param_pw['flag_max'] = True
            param_pw['next_turn_value'] = value_ps[:,:,score_state_ps] ## player B throws in next turn
            param_pw['game_end_value'] = value_win_pw ## end game state A win
            
            ## player B
            param_ps['score_state'] = score_state_ps  
            param_ps['token_state'] = 0     
            param_ps['state_len_vector'] = state_len_vector_ps
            param_ps['state_action'] = state_action_ps
            param_ps['state_value'] = state_value_ps
            param_ps['state_action_update'] = state_action_update_ps
            param_ps['state_value_update'] = state_value_update_ps
            param_ps['action_diff'] = action_diff_ps
            param_ps['value_relerror'] = value_relerror_ps    
            ## maximize player B's win_prob = minimize player A's win_prob
            param_ps['flag_max'] = False
            param_ps['next_turn_value'] = value_pw[:,score_state_pw,:] ## player A throws in next turn
            param_ps['game_end_value'] = value_win_ps ## end game state B win     
            
            ## optimize A and B iteratively
            for round_index_zsgtwoplayers in range(iteration_round_limit_zsgtwoplayers):
                ## print('## optimize two players round = {} ##'.format(round_index_zsgtwoplayers))
                ## iterate at least once for each player
                
                #### optimize A policy ####
                value_pw_state_old = value_pw[tokens_pw, score_state_pw,score_state_ps] ## starting value 0
                value_ps_state_old = value_ps[tokens_pw, score_state_pw,score_state_ps] ## starting value 0                
                for round_index in range(iteration_round_limit_singleplayer_policy):                    
                    
                    ## policy evaluation
                    tpe1 = time.time()                
                    ## use the initial prob_turn_transit_pa value for round_index=0
                    if (round_index >=0):
                        prob_turn_transit_pw = h.solve_turn_transit_probability_fast_token(score_state=score_state_pw,state_action=state_action_pw,available_tokens=tokens_pw,prob_grid_normalscore_nt=prob_grid_normalscore_nt_pw,prob_grid_doublescore_nt=prob_grid_doublescore_nt_pw,prob_grid_bullscore_nt=prob_grid_bullscore_nt_pw,prob_bust_dic_nt=prob_bust_dic_nt_pw,prob_grid_normalscore_t=prob_grid_normalscore_t,prob_grid_doublescore_t=prob_grid_doublescore_t,prob_grid_bullscore_t=prob_grid_bullscore_t,prob_bust_dic_t=prob_bust_dic_t)
                    ## player B is fixed, use stored value
                    [value_state_pw, value_state_ps] = zsg_policy_evaluation_tokens(value_pw, value_ps, tokens_pw, score_state_pw, score_state_ps, prob_turn_transit_pw, prob_turn_transit_ps)
                    value_pw[tokens_pw, score_state_pw, score_state_ps] = value_state_pw
                    value_ps[tokens_pw, score_state_pw, score_state_ps] = value_state_ps
                    tpe2 = time.time()
                    t_policy_evaluation += (tpe2-tpe1) 

                    #### policy improvement for player A ####
                    tpi1 = time.time()
                    param_pw['round_index'] = round_index
                    [max_action_diff_pw, max_value_relerror_pw] = zsg_policy_improvement_tokens(param_pw)
                    tpi2 = time.time()
                    t_policy_improvement += (tpi2 - tpi1)
                    if (max_action_diff_pw < 1):
                        break    
                    if (max_value_relerror_pw < iteration_relerror_limit_singleplayer_policy):
                        break
        
                optimal_action_index_dic_pw[score_state_pw][score_state_ps] = state_action_pw
                optimal_value_dic_pw[score_state_pw][score_state_ps] = state_value_pw
                prob_turn_transit_dic_pw[score_state_pw][score_state_ps] = prob_turn_transit_pw
                num_iteration_record_pw[tokens_pw,score_state_pw, score_state_ps] = round_index + 1
                #### done optimize player A
                
                ## check optimality
                value_pw_optW[tokens_pw, score_state_pw,score_state_ps] = value_pw[tokens_pw, score_state_pw,score_state_ps]
                value_ps_optW[tokens_pw, score_state_pw,score_state_ps] = value_ps[tokens_pw, score_state_pw,score_state_pw]
                max_zsgvalue_relerror = max([np.abs(value_pw_state_old-value_pw[tokens_pw, score_state_pw,score_state_ps]), np.abs(value_ps_state_old-value_ps[tokens_pw, score_state_pw,score_state_ps])])
                #print('A:max_zsgvalue_relerror={}'.format(max_zsgvalue_relerror))      
                if (max_zsgvalue_relerror < iteration_relerror_limit_zsgtwoplayers):
                    break


                #### optimize B policy ####
                value_pw_state_old = value_pw[tokens_pw, score_state_pw,score_state_ps] ## starting value 0
                value_ps_state_old = value_ps[tokens_pw, score_state_pw,score_state_ps] ## starting value 0                
                for round_index in range(iteration_round_limit_singleplayer_policy):                    
                    
                    ## policy evaluation
                    tpe1 = time.time()
                    ## player A is fixed, only need to compute once
                    if (round_index >=0):
                        prob_turn_transit_ps = h.solve_turn_transit_probability_fast_token(score_state=score_state_ps,state_action=state_action_ps,available_tokens=0,prob_grid_normalscore_nt=prob_grid_normalscore_nt_ps,prob_grid_doublescore_nt=prob_grid_doublescore_nt_ps,prob_grid_bullscore_nt=prob_grid_bullscore_nt_ps,prob_bust_dic_nt=prob_bust_dic_nt_ps,prob_grid_normalscore_t=prob_grid_normalscore_t,prob_grid_doublescore_t=prob_grid_doublescore_t,prob_grid_bullscore_t=prob_grid_bullscore_t,prob_bust_dic_t=prob_bust_dic_t)
                    [value_state_pw, value_state_ps] = zsg_policy_evaluation_tokens(value_pw, value_ps, tokens_pw, score_state_pw, score_state_ps, prob_turn_transit_pw, prob_turn_transit_ps)
                    value_pw[tokens_pw, score_state_pw, score_state_ps] = value_state_pw
                    value_ps[tokens_pw, score_state_pw, score_state_ps] = value_state_ps
                    tpe2 = time.time()
                    t_policy_evaluation += (tpe2-tpe1) 

                    #### policy improvement for player B ####
                    tpi1 = time.time()
                    param_ps['round_index'] = round_index
                    [max_action_diff_ps, max_value_relerror_ps] = zsg_policy_improvement_tokens(param_ps)
                    tpi2 = time.time()
                    t_policy_improvement += (tpi2 - tpi1)
                    if (max_action_diff_ps < 1):
                        break    
                    if (max_value_relerror_ps < iteration_relerror_limit_singleplayer_policy):
                        break
        
                optimal_action_index_dic_ps[score_state_pw][score_state_ps] = state_action_ps
                optimal_value_dic_ps[score_state_pw][score_state_ps] = state_value_ps
                prob_turn_transit_dic_ps[score_state_pw][score_state_ps] = prob_turn_transit_ps
                num_iteration_record_ps[tokens_pw, score_state_pw, score_state_ps] = round_index + 1
                #### done optimize player B
        
                ## check optimality
                value_pw_optS[tokens_pw, score_state_pw,score_state_ps] = value_pw[tokens_pw, score_state_pw,score_state_ps]
                value_ps_optS[tokens_pw, score_state_pw,score_state_ps] = value_ps[tokens_pw, score_state_pw,score_state_ps]
                max_zsgvalue_relerror = max([np.abs(value_pw_state_old-value_pw[tokens_pw, score_state_pw,score_state_ps]), np.abs(value_ps_state_old-value_ps[tokens_pw, score_state_pw,score_state_ps])])
                #print('B:max_zsgvalue_relerror={}'.format(max_zsgvalue_relerror))
                if (max_zsgvalue_relerror < iteration_relerror_limit_zsgtwoplayers):
                    break
            
            #### done optimize A and B iteratively
            value_pw[tokens_pw,score_state_pw,score_state_ps] = 0.5*(value_pw_optW[tokens_pw, score_state_pw,score_state_ps]+value_pw_optS[tokens_pw, score_state_pw,score_state_ps])
            value_ps[tokens_pw,score_state_pw,score_state_ps] = 0.5*(value_ps_optW[tokens_pw, score_state_pw,score_state_ps]+value_ps_optS[tokens_pw, score_state_pw,score_state_ps])
            iteration_round_zsgtwoplayers[tokens_pw,score_state_pw,score_state_ps] = round_index_zsgtwoplayers + 1
            #print('optimize A and B iteratively in time={} seconds'.format(time.time()-t_opt_twoplayers_begin))
        
        #### finish a column        
        #if (score_state_pb%20==0 or score_state_pb==2):
        #    print('#### score_state_pb={}, time={}'.format(score_state_pb, time.time()-t_scoreloop_begin))

    ## computation is done
    t2 = time.time()
    print('solve_zsg_opt_{}e{}_fix_{}e{} in {} seconds'.format(name_ps,epsilon_ps, name_pw, epsilon_pw, t2-t1))
    print('t_policy_evaluation  = {} seconds'.format(t_policy_evaluation))
    print('t_policy_improvement = {} seconds'.format(t_policy_improvement))
    print('t_other = {} seconds'.format(t_other))   
    #print('value_pa {} '.format(value_pa))
    #print('value_pb {} '.format(value_pb))
        
    
    result_dic = {'info':info, 'optimal_action_index_dic_pw':optimal_action_index_dic_pw, 'optimal_action_index_dic_ps':optimal_action_index_dic_ps, 'value_pw':value_pw, 'value_ps':value_ps, 'value_pw_optW':value_pw_optW, 'value_pw_optS':value_pw_optS, 'value_ps_optW':value_ps_optW, 'value_ps_optS':value_ps_optS,  'optimal_value_dic_pw':optimal_value_dic_pw, 'optimal_value_dic_ps':optimal_value_dic_ps, 'iteration_round_zsgtwoplayers':iteration_round_zsgtwoplayers, 'num_iteration_record_pw':num_iteration_record_pw, 'num_iteration_record_ps':num_iteration_record_ps}
    if (result_dir is not None):
        ft.dump_pickle(result_filename, result_dic)
        print('save {}'.format(result_filename))
        ft.dump_pickle(result_value_filename, {'info':info, 'value_pw':value_pw, 'value_ps':value_ps, 'iteration_round_zsgtwoplayers':iteration_round_zsgtwoplayers})
        print('save {}'.format(result_value_filename))
        return 'save'
    else:
        return result_dic

