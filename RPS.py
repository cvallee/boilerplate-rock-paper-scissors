# The example function below keeps track of the opponent's history and plays whatever the opponent played two plays ago. It is not a very good player so you will need to change the code to pass the challenge.
'''
This is my code for solving the freeCodeCamp Rock Paper Scissors challenge.

I am slighlty confused because everytime I run the for lines of play() in the main.py file I have > 60% win rate against every player,
but I always fail when running the `main(module='test_module', exit=False)` line.
Anyway, this is the best I can do.
'''

import random
import numpy as np

# Initialise the Q-matrix
Q1 = np.zeros((9, 3))
Q2 = np.zeros((9, 3))
epsilon=0.1
LR=0.98
GAMMA=0.9

def win_rate(player_history, opponent_history):
    wins = 0
    ties = 0
    defeats = 0
    for p, o in zip(player_history, opponent_history):
        if p == o:
            ties += 1
        elif (p == "R" and o == "S") or (p == "P" and o == "R") or (p == "S" and o == "P"):
            wins += 1
        else:
            defeats += 1
    if defeats == 0:
        rate = 1
    else:
        rate = wins / (wins+defeats)
    return rate


def most_frequent_counter(player_history, n=10):
    win = {"P": "S", "R": "P", "S": "R"}
    most_frequent = max(set(player_history[-10:]), key=player_history[-10:].count)
    prediction = win[most_frequent]
    guess = win[prediction]
    return guess
    

def sequence_counter(player_history, n=5):
    win = {"P": "S", "R": "P", "S": "R"}
    history = "".join(player_history)
    sequence = history[-(n-1):]
    potential_play = [sequence + a for a in ["R", "P", "S"]]
    counts = [history.count(p) for p in potential_play]
    prediction = potential_play[max(enumerate(counts),key=lambda x: x[1])[0]][-1]
    guess = win[prediction]
    return guess


# Get reward from last guess
def get_reward(choice_player1, choice_player2):
    
    if (choice_player1 == "R" and choice_player2 == "S") or (choice_player1 == "P" and choice_player2 == "R") or (choice_player1 == "S" and choice_player2 == "P"):
        reward = 2
    elif choice_player1 == choice_player2:
        reward = 0
    else:
        reward = -1
    return reward


def update_q(Q, reward, history):
    
    action_to_int={"R":0, "P":1, "S":2}
    int_to_action={0:"R", 1:"P", 2:"S"}
    state_to_int={"RR":0, "RP": 1, "RS":2, "PR":3, "PP": 4, "PS":5, "SR":6, "SP": 7, "SS":8}
   
    state = state_to_int["".join(history[-2:])]
    if np.random.uniform(0,1) < epsilon:
        action = action_to_int[random.choice(["R", "P", "S"])]
    else:
        action = np.argmax(Q[state, :])
    
    next_state = state_to_int[history[-1]+int_to_action[action]]
    
    Q[state, action] = Q[state, action] + LR * (reward+GAMMA*np.max(Q[next_state, :]) - Q[state, action])
    
    return Q, Q[state, action], int_to_action[action]


def player(prev_play, Q1=Q1, Q2=Q2, my_history=[], opponent_history=[], counter=[0], strategy=[0]):
    
    # Count number of plays
    counter[0] += 1
    # Append opponent play
    if prev_play:
        opponent_history.append(prev_play)
    # Set up winning rules
    win = {"P": "S", "R": "P", "S": "R"}
    # First 2 plays are random
    if counter[0] <= 2:
        guess = random.choice(["R", "P", "S"])
    else:
        # Check win_rate every 10 plays
        if counter[0] % 20:
            rate = win_rate(my_history, opponent_history)
            # Change strategy if win rate is below 57.5%
            if rate < 0.575:
                strategy[0] = ((strategy[0] + 1) % 4)
        # Update Q tables with both player and opponent histories
        reward1 = get_reward(my_history[-1], opponent_history[-1])
        Q1, odds1, pred1 = update_q(Q1, reward1, my_history)
        reward2 = get_reward(opponent_history[-1], my_history[-1])
        Q2, odds2, pred2 = update_q(Q2, reward2, opponent_history)
        if strategy[0] == 0:
            guess = sequence_counter(opponent_history, n=4)
        elif strategy[0] == 1:
            guess = most_frequent_counter(my_history, n=10)
        elif strategy[0] == 2:
            guess = pred1
        else:
            guess = win[pred2]
        
    my_history.append(guess)
        
    return guess
