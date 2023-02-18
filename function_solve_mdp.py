# -*- coding: utf-8 -*-
"""function_solve_mdp.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1VzuADHPjAjSfRqr0c6N5fln6-hM8G46P
"""

def generate_policy_transition_probabilities(states,policy,transition_probabilities):
  policy_tps = np.zeros([501,501])

  for s in states: 
    pi = policy[s]
    policy_tps[s] = transition_probabilities[pi][s]

  return policy_tps

def initialize_policy(rand=False):
  if rand==False:
    # policy = ['SI3-c-o' for s in states]
    # policy[1] = 'D1-c-i'
    policy = ['SI1-c-o' for s in states]
    policy[1] = 'D1-c-i'
  else:
    policy = ['SI1-c-o' for s in states]
    policy[1] = 'D1-c-i'
    # i=0
    # policy = [random.choice(list(actions)) for s in states]
    # while is_proper_policy(policy,transition_probabilities) == False:
    #   policy = [random.choice(list(actions)) for s in states]
    #   i+=1
    #   if i%1000==0:
    #     print(f"{i} iterations...")
    # print(f"Took {i} iterations to find a proper policy.")
  return policy


def evaluate_policy(policy,transition_probabilities,costs):

  # Initialize transition probabilities for policy
  policy_tps = generate_policy_transition_probabilities(states,policy,transition_probabilities)

  # Set value of all states to one except for v(0)=0, initialize stopping condition
  v_old = np.ones(501)
  v_old[0]=0 
  v_sub = (np.zeros(501) + 1000) - v_old 
  i = 0

  while max(v_sub) > 0.01:
    v_new = costs + policy_tps.dot(v_old)
    v_sub = np.abs(v_new - v_old)
    v_old = v_new
    i+=1
    if max(v_new) > 510:
      print(f'Terminated after {i} iterations. \nState {np.argmax(v_new)} has value of {max(v_new)}. \nImproper policy - will not converge.') 
      break 

  print(f"(Policy evaluation converged in {i} iterations.)")
  return v_new

def policy_selection(states,actions,transition_probabilities,costs,values):
  
  policy_temp = {s:('',100000000) for s in states}

  for a in actions:
    a_tps = transition_probabilities[a]
    v_est = costs + a_tps.dot(values)

    for s in states:
      if v_est[s] < policy_temp[s][1]:
        policy_temp[s]=(a,v_est[s])
  
  policy=[policy_temp[s][0] for s in states]

  return policy

def policy_iteration(states,actions,transition_probabilities,costs,max_iter=1000,rand_init=False):
  
  # Initialize Policy
  policy_old = initialize_policy(rand=rand_init)
  policy_tps = generate_policy_transition_probabilities(states,policy_old,transition_probabilities)

  for i in range(0,max_iter):
    values = evaluate_policy(policy_old,transition_probabilities,costs)
    policy_new = policy_selection(states,actions,transition_probabilities,costs,values)
    if policy_new == policy_old:
      print(f"Policy iteration converged in {i} iterations.")
      return policy_new
    policy_old=policy_new