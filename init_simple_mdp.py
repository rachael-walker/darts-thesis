import init_load_board as b 
import numpy as np

#---------------------------------------------------------------------------------------------
############################### ---- Initialize States ---- ##################################
#---------------------------------------------------------------------------------------------

states_indexes = list(range(0,501,1))
states = {}

for i in states_indexes:
  # The zero index will correspond to a state value of zero (absorbing states)
  if i==0:
    states[i]=i
  # All other indexes correspond to a value i+1 since the 1 state was removed (not reachable)
  else:
    states[i]=i+1 


#---------------------------------------------------------------------------------------------
############################# ---- Initialize Throw Actions ---- #############################
#---------------------------------------------------------------------------------------------

actions = {}

# For i in segements of board 1-20 populate actions for doubles, triples, singles
for i in range(0,20):

  for orientation in b.board_segment_orientations:

    for region in b.board_regions:

      for distance in ['o','m','i']:

        #print(i, board_segments[i],math.degrees(board_segment_orientations[orientation][i]))
        name = region + str(b.board_segments[i]) + '-' + orientation + '-' + distance

        x = b.board_region_distances[region][distance] * np.sin(b.board_segment_orientations[orientation][i]) 
        y = b.board_region_distances[region][distance] * np.cos(b.board_segment_orientations[orientation][i]) 

        x_int = round(x)
        y_int = round(y)

        actions[name] = {"coords":(x,y),
                         "gaussian_centroid":(x_int,y_int),
                         "value":b.board_segments[i]*b.board_regions[region]}

# populate single bullseye
for orientation in b.board_segment_orientations_bullseyes:

  name = 'SB' + '-' + orientation 

  x = b.board_region_distances['SB']['m'] * np.sin(b.board_segment_orientations_bullseyes[orientation]) 
  y = b.board_region_distances['SB']['m'] * np.cos(b.board_segment_orientations_bullseyes[orientation]) 
  
  x_int = round(x)
  y_int = round(y)  

  actions[name] = {"coords":(x,y),
                         "gaussian_centroid":(x_int,y_int),
                         "value":25}

# populate double bullseye
actions['DB-c'] = {"coords":(0,0),
                         "gaussian_centroid":(0,0),
                         "value":50}

#---------------------------------------------------------------------------------------------
############################ ---- Initialize Result Values ---- ##############################
#---------------------------------------------------------------------------------------------

result_values = {}

# For i in segements of board 1-20

for i in range(1,21):

  # Add S{i} (single)
  s_name = "S" + str(i)
  result_values[s_name] = i 

  # Add D{i} (double)
  d_name = "D" + str(i)
  result_values[d_name] = i*2 

  # Add T{i} (triple)
  t_name = "T" + str(i)
  result_values[t_name] = i*3 

# Add SB
sb_name = "SB"
result_values[sb_name] = 25 

# Add DB
db_name = "DB"
result_values[db_name] = 50 

#---------------------------------------------------------------------------------------------
############################ ---- Initialize Token Actions ---- ##############################
#---------------------------------------------------------------------------------------------

# Token Actions 

token_actions = {}

board_segments = [20, 1, 18 , 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5]

regions = ['S','D','T']

for r in regions: 

    # Radial Segments 
    for i in range(0,20): 

        name = r + str(str(board_segments[i]))

        x = b.board_region_distances['SO']['m'] * np.sin(b.board_segment_orientations['c'][i]) 
        y = b.board_region_distances['SO']['m'] * np.cos(b.board_segment_orientations['c'][i]) 

        token_actions[name] = {
            'coords':(x,y),
            'gaussian_centroid': (round(x),round(y)),
            'value': result_values[name]
        }
    
    # SB
    token_actions['SB'] = {"coords":(0.0, 11.125),
                        "gaussian_centroid":(0, 11),
                        "value":25}

    # DB 
    token_actions['DB'] = {"coords":(0,0),
                            "gaussian_centroid":(0,0),
                            "value":50}

#---------------------------------------------------------------------------------------------
############################ ---- Initialize Action Objects---- ##############################
#---------------------------------------------------------------------------------------------

all_actions = actions.copy()
all_actions = all_actions.update(token_actions)

a_throw_list = []
a_token_list = []

for a in actions:
    a_throw_list.append(a)

for a in token_actions:
    a_token_list.append(a)

a_list = a_throw_list + a_token_list
throw_num = len(a_throw_list)


#---------------------------------------------------------------------------------------------
################################# ---- Initialize Costs ---- #################################
#---------------------------------------------------------------------------------------------

# All states have a cost of 1 except for the absorbing state
# Only have a size of 501 since the 1 state has been removed (not reachable)

costs=np.ones(501)
costs[0]=0