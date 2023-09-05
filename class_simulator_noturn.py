import function_helpers_singleplayer as h
import function_init_board as fb
import function_init_simple_mdp as imdp
import numpy as np
import time


class Simulator :

    def __str__(self):
        return f"Simulator for {self.name_pa} with epsilon ({self.epsilon})"

    def __init__(self, player_num=10, epsilon=1):

        # Initialize Player Parameters 
        self.player_num = player_num
        self.epsilon = epsilon
        self.name_pa = 'player{}'.format(player_num)

        # Extract Transition Probabilities 
        [aiming_grid, prob_grid_normalscore_nt, prob_grid_singlescore_nt, prob_grid_doublescore_nt, prob_grid_triplescore_nt, prob_grid_bullscore_nt] = h.load_aiming_grid(self.name_pa, epsilon=epsilon, data_parameter_dir=fb.data_parameter_dir, grid_version='custom_no_tokens')
        [aiming_grid, prob_grid_normalscore_t, prob_grid_singlescore_t, prob_grid_doublescore_t, prob_grid_triplescore_t, prob_grid_bullscore_t] = h.load_aiming_grid('t', data_parameter_dir=fb.data_parameter_dir, grid_version='custom_tokens')

        transition_probs = np.zeros((len(aiming_grid),63))

        for action in range(imdp.throw_num): 

            transition_probs[action][0] = prob_grid_normalscore_nt[action][0]
            transition_probs[action][1:21] = prob_grid_singlescore_nt[action][:]
            transition_probs[action][21:41] = prob_grid_doublescore_nt[action][:]
            transition_probs[action][41:61] = prob_grid_triplescore_nt[action][:]
            transition_probs[action][61:63] = prob_grid_bullscore_nt[action][:]

        for action in range(imdp.throw_num,len(aiming_grid)): 

            transition_probs[action][0] = prob_grid_normalscore_t[action][0]
            transition_probs[action][1:21] = prob_grid_singlescore_t[action][:]
            transition_probs[action][21:41] = prob_grid_doublescore_t[action][:]
            transition_probs[action][41:61] = prob_grid_triplescore_t[action][:]
            transition_probs[action][61:63] = prob_grid_bullscore_t[action][:]

        result_list = []
        result_list.append('miss')

        for i in range(1,21):
            result_list.append('S{}'.format(i))

        for i in range(1,21):
            result_list.append('D{}'.format(i))

        for i in range(1,21):
            result_list.append('T{}'.format(i))

        result_list.append('SB')
        result_list.append('DB')

        # Initialize Transition Probabilities 
        self.transition_probs = transition_probs
        self.result_list = result_list

        # Initialize Optimal Policy and Values
        [optimal_value, optimal_action_index] = h.solve_dp_noturn_tokens(aiming_grid, prob_grid_normalscore_nt, prob_grid_normalscore_t, tokens = 9, prob_grid_doublescore=prob_grid_doublescore_nt, prob_grid_bullscore=prob_grid_bullscore_nt, prob_grid_doublescore_dic=None, prob_grid_doublescore_t=prob_grid_doublescore_t, prob_grid_bullscore_t=prob_grid_bullscore_t, prob_grid_doublescore_dic_t=None)

        self.optimal_value = optimal_value
        self.optimal_action_index = optimal_action_index

    def spot_darts(self,start_score=501,spotted_darts=1):
        spotted_darts = spotted_darts
        tokens = 0
        start_score = start_score

        points_gained = 0 

        optimal_policy = self.optimal_action_index

        for i in range(spotted_darts):
            
            best_throw_guess = optimal_policy[tokens][start_score]
            
            action_result = np.random.choice(list(range(63)), 1, p = self.transition_probs[best_throw_guess])[0]
            result_name = self.result_list[action_result]

            if result_name=='miss':
                result_value = 0
            else:
                result_value = imdp.result_values[result_name]

            points_gained += result_value

        return points_gained
    
    def simulate_game(self, starting_score=501, starting_credits=0, spot_darts=None, spot_points=None):
        path = []

        s = starting_score 
        t = starting_credits

        if spot_points:
            s = s - spot_points 
        
        if spot_darts:
            score_gained = self.spot_darts(start_score=s, spotted_darts=spot_darts)
            s = s - score_gained

        while s > 1:

            path.append((s,t))

            optimal_action =  self.optimal_action_index[t,s]
            is_token = self.optimal_action_index[t,s] >= imdp.throw_num

            action_result = np.random.choice(list(range(63)), 1, p = self.transition_probs[optimal_action])[0]
            result_name = self.result_list[action_result]

            if result_name=='miss':
                result_value = 0
            else:
                result_value = imdp.result_values[result_name]

            # Cannot go bust 
            if s > 60: 

                if is_token: 
                    t = t - 1 

                s = s - result_value

            # Can go bust 
            else: 

                if is_token: 
                    t = t - 1 

                # Check Out
                if s - result_value == 0  and 'D' in result_name:
                    s = s - result_value
                    path.append((s,t))
                
                # Go Bust 
                elif s - result_value < 2 :
                    continue 

                # Continue 
                else:
                    s = s - result_value

        return path
    
    # Run Simulation
    def run_simulation(self, iterations, starting_score=501, starting_credits=0, spot_darts=None, spot_points=None):
        simulation_paths = []
        start = time.time()

        for iter in range(iterations):
            if iter % 10000 == 0 and iter > 0:
                print(f"Ran {iter} iterations",'-',f"{time.time()-start:.3f}s")
            simulation_paths.append(self.simulate_game(starting_score, starting_credits, spot_darts=spot_darts, spot_points=spot_points))

        print(f"Ran in {time.time()-start:.3f}s.")

        return simulation_paths

    def get_transition_probs(self):
        return self.transition_probs
        