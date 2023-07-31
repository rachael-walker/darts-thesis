import pandas as pd
import init_simple_mdp as imdp
import math


class Handicap:

    def __str__(self):
        return f"Handicap Generator Object "

    def __init__(self, results_file_name='player10_results.csv', epsilon_list = list(range(1,10))):

        self.df = pd.read_csv(results_file_name)
        try:
            self.df = self.df[['epsilon','tokens','score','remaining_throws','score_gained','optimal_value','optimal_policy']]
        except: 
            self.df = self.df[['epsilon','tokens','score','optimal_value','optimal_policy']]


        # Map to policy action names 
        a_map = {i:imdp.a_list[i] for i in range(len(imdp.a_list))}
        self.df['optimal_policy_name'] = self.df['optimal_policy'].map(a_map)

        self.epsilon_list = epsilon_list

        self.spot_point_heuristic_df = self.init_spot_point_heuristic_handicap()
        self.dynamic_credit_lookup = self.init_dynamic_credits_handicap()
        self.spot_point_optimal_lookup = self.init_spot_point_optimal_handicap()

    def init_spot_point_heuristic_handicap(self):

        expected_turns_list = []
        for epsilon in self.epsilon_list: 
            try:
                t = self.df.loc[(self.df.score==501) & (self.df.epsilon==epsilon) & (self.df.tokens==0) & (self.df.remaining_throws==3)].optimal_value.values[0]
            except: 
                t = self.df.loc[(self.df.score==501) & (self.df.epsilon==epsilon) & (self.df.tokens==0)].optimal_value.values[0]

            expected_turns_list.append(t)

        ppd_list = [501 / (expected_turns_list[i]*3) for i in range(len(expected_turns_list))]

        ssl_list = []

        for i in range(len(ppd_list)):

            for j in range(i,len(ppd_list)):

                hppd = ppd_list[i]
                lppd = ppd_list[j]

                ssl = round(501*(lppd/hppd))

                ssl_list.append([self.epsilon_list[i],self.epsilon_list[j],ssl])

        #ssl_list

        ssl_list_full = []

        for ssl in ssl_list: 

            th1 = self.df.loc[(self.df.score==501) & (self.df.epsilon==ssl[0]) & (self.df.tokens==0)].optimal_value.values[0] #* 3
            th2 = self.df.loc[(self.df.score==ssl[2]) & (self.df.epsilon==ssl[1]) & (self.df.tokens==0)].optimal_value.values[0] #* 3

            ssl_list_full.append([ssl[0],ssl[1],ssl[2],th1,th2, th2-th1])

        spot_point_df = pd.DataFrame(ssl_list_full,columns=['p1_epsilon','p2_epsilon','p2_starting_score','p1_expected_turns','p2_expected_turns','diff_expected_turns'])

        return spot_point_df

    def linear_interpolation(self,x1,x2,y1,y2,y):
        x = x1 - ((y1-y)/(y1-y2))*(x1-x2)
        return x 

    def init_dynamic_credits_handicap(self):
        # Get dataframe only for score values at 501 
        df_501 = self.df.loc[self.df.score==501]
        try: 
            df_501 = df_501.loc[df_501.remaining_throws==3]
        except: 
            pass 

        # Create dataframes for values at 501 for each token value 
        epsilon_dfs = []

        for e in range(len(self.epsilon_list)):
            epsilon = self.epsilon_list[e]
            temp = df_501.loc[df_501.epsilon==epsilon].set_index('tokens')['optimal_value'].copy()
            epsilon_dfs.append(temp)
       
        handicap_vals = {}

        for i in range(len(self.epsilon_list)):

            handicap_vals[self.epsilon_list[i]] = {}

        for i in range(len(self.epsilon_list)):

            stronger_no_token = epsilon_dfs[i][0]

            for j in range(len(self.epsilon_list)):

                if self.epsilon_list[i] <= self.epsilon_list[j]: 

                    df_j = epsilon_dfs[j]

                    for b in range(1,len(df_j)):

                        x1 = b-1
                        x2 = b 
                        y1 = df_j[b-1]
                        y2 = df_j[b]

                        if (stronger_no_token <= y1) & (stronger_no_token >= y2):

                            handicap = self.linear_interpolation(x1,x2,y1,y2,stronger_no_token)

                            handicap_vals[self.epsilon_list[i]][self.epsilon_list[j]] = handicap

        return handicap_vals

    def init_spot_point_optimal_handicap(self):

        df0 = self.df.loc[self.df.tokens==0].copy()
        try: 
            df0 = df0.loc[df0.remaining_throws==3].copy()
        except: 
            pass 

        # Create dataframes for values at 501 for each token value 
        epsilon_dfs = []

        for e in range(len(self.epsilon_list)):
            epsilon = self.epsilon_list[e]
            temp = df0.loc[df0.epsilon==epsilon].set_index('score')['optimal_value'].copy()
            epsilon_dfs.append(temp)
        
        sp_handicap_vals = {}

        for i in range(len(self.epsilon_list)):

            sp_handicap_vals[self.epsilon_list[i]] = {}

        for i in range(len(self.epsilon_list)):

            stronger_no_token = epsilon_dfs[i][501] #*3

            for j in range(len(self.epsilon_list)):

                #if j <= i:
                if self.epsilon_list[i] <= self.epsilon_list[j]: 

                    df_j = epsilon_dfs[j] #*3

                    for s in range(3,502):

                        x1 = s-1
                        x2 = s 
                        y1 = df_j[s-1] 
                        y2 = df_j[s] 

                        # print(f"\t {i}-{j}-{s}")
                        # print(stronger_no_token,y1,y2)

                        if (stronger_no_token >= y1) & (stronger_no_token <= y2):

                            handicap = self.linear_interpolation(x1,x2,y1,y2,stronger_no_token)

                            sp_handicap_vals[self.epsilon_list[i]][self.epsilon_list[j]] = handicap
                        
        return sp_handicap_vals

    
    def get_spot_point_heuristic_handicap(self,epsilon_strong,epsilon_weak):

        if epsilon_strong > epsilon_weak:
            print('Epsilon values are in the wrong order. Stronger player should be first.')
            return -1 

        start_score_handicap = self.spot_point_heuristic_df.loc[(self.spot_point_heuristic_df.p1_epsilon == epsilon_strong) & (self.spot_point_heuristic_df.p2_epsilon == epsilon_weak)]['p2_starting_score'].values
        spot_points = 501 - start_score_handicap[0].round()

        return int(spot_points) 

    def get_spot_point_optimal_handicap(self,epsilon_strong,epsilon_weak):

        if epsilon_strong > epsilon_weak:
            print('Epsilon values are in the wrong order. Stronger player should be first.')
            return -1 

        start_score_handicap = self.spot_point_optimal_lookup[epsilon_strong][epsilon_weak]
        spot_points = 501 - start_score_handicap
        spot_points = spot_points.round()

        return int(spot_points)

    def get_dynamic_credits_handicap(self,epsilon_strong,epsilon_weak):

        if epsilon_strong > epsilon_weak:
            print('Epsilon values are in the wrong order. Stronger player should be first.')
            return -1 

        tokens = self.dynamic_credit_lookup[epsilon_strong][epsilon_weak]
        tokens_lower = math.floor(tokens)
        tokens_higher = math.ceil(tokens)
        threshold_higher_multiplier = (tokens-tokens_lower)

        return tokens, tokens_lower, tokens_higher, threshold_higher_multiplier