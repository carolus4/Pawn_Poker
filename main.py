import numpy as np
from torch import nn


import random
import torch
import time
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from collections import deque


#  from check_submission import check_submission
from game_mechanics import (
    ChooseMoveCheckpoint,
    PokerEnv,
    checkpoint_model,
    choose_move_randomly,
    human_player,
    load_network,
    play_poker,
    save_network,
)

# Global Vars
random_env = PokerEnv(choose_move_randomly, 0)
s_size = random_env.observation_space.shape[0]
a_size = random_env.action_space.n

just_play = False

training_episodes = 0 if just_play else 1000

poker_hyperparameters = {
    "h1_size": 200,
    "n_training_episodes": training_episodes,
    "n_evaluation_episodes": 10,
    "print_every": 100,
    "max_t": 100,
    "gamma": 1.0,
    "lr": 1e-3,
    "state_space": s_size,
    "action_space": a_size
}

# ARCHITECTURE
# We assume the following architecture:
# 1.FCN         state_space  (s_size = ) [+ --> ELU]
# 2.FCN     --> intermediate (h1_size = 200) [arbitrary] [+ --> ReLU]
# 3.FCN     --> intermediate (h2_size = 32) [arbitrary]
# 4.Softmax --> action_space (probability distribution over a_space)

class Policy(nn.Module):
    def __init__(self, s_size, a_size, h1_size):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(s_size, h1_size)
        self.fc2 = nn.Linear(h1_size, a_size)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=-1)

    def act(self, state):
        state = torch.from_numpy(state[0]).float().unsqueeze(0)
        probs = self.forward(state)
        m = Categorical(probs)

        action = m.sample()
        return action.item(), m.log_prob(action)


def debug():
    print()
    initial_state = random_env.reset()

    print("_____OBSERVATION SPACE_____")
    print("The state space is: ", s_size)
    print("Initial observation: ", initial_state)
    print()

# _____OBSERVATION SPACE_____
# The state space is:  72
# Initial observation:  (array([0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    #    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
    #    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    #    0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.,
    #    0., 0., 0., 0.], dtype=float32), 0, False, {'legal_moves': array([0, 1, 2])})

    print("_____DEBUG 1-STEP_____")

    debug_policy = Policy(s_size, a_size, 32)
    debug_action, debug_logprob = debug_policy.act(random_env.reset())
    print("Debug action: ", debug_action)
    print("Debug logprob: ", debug_logprob)
    print()

def get_policy():
    print("_____SETUP POLICY_____")

    # TODO - Try to load, except new policy

    poker_policy = Policy(
        poker_hyperparameters["state_space"],
        poker_hyperparameters["action_space"],
        poker_hyperparameters["h1_size"]
    )

    print("NEW Poker Play policy")
    print(poker_policy)
    print()

    return poker_policy

# _____SETUP POLICY_____
# NEW Poker Play policy
# Policy(
#   (fc1): Linear(in_features=72, out_features=200, bias=True)
#   (fc2): Linear(in_features=200, out_features=4, bias=True)
# )

def get_optimizer(policy):
    print("_____SETUP OPTIMIZER_____")
    poker_optimizer = optim.Adam(policy.parameters(), 
        lr= poker_hyperparameters["lr"] )

    print(poker_optimizer)
    print()
    return poker_optimizer

TEAM_NAME = "Pawn"  # <---- Enter your team name here!
assert TEAM_NAME != "Team Name", "Please change your TEAM_NAME!"


def train(policy, optimizer, n_training_episodes, max_t, gamma,
        print_every ) -> nn.Module:
    """
    Returns:
        pytorch network
    """

    scores_deque = deque(maxlen=print_every)
    scores = []

    start = time.time()

    for i_episode in range(1, n_training_episodes + 1):
        saved_log_probs = []
        rewards = []
        state = random_env.reset()
        # TODO - train against other envs

        for t in range(max_t):
            action, logprob = policy.act(state)
            saved_log_probs.append(logprob)
            state, reward, done, _ = random_env.step(action)
            rewards.appen(reward)
            if done:
                break 
                # Note - I dont think this setup ever breaks?
        
        scores_deque.append(sum(rewards))
        scores.append(sum(rewards))

        discounts = [gamma ** i for i in range(len(rewards))]
        # Is this correct? Are games independent? 
        # Should they be??

    # Below is an example of how to checkpoint your model and
    # the train against these checkpoints
    # env = PokerEnv(choose_move_randomly)
    # model = ...
    # Train model in env .........
    # checkpoint_model(model, "checkpoint1")
    # env = PokerEnv(
    #     ChooseMoveCheckpoint("checkpoint1", choose_move).choose_move)
    # Train model in env .........
    return nn.Linear(1, 1)


def choose_move(state: np.ndarray, legal_moves: np.ndarray,
                neural_network: nn.Module) -> int:
    """Called during competitive play. It acts greedily given current state of the board and your
    network. It returns a single move to play.

    Args:
         state: The state of poker game. shape = (72,)
         legal_moves: Legal actions on this turn. Subset of {0, 1, 2, 3}
         neural_network: Your pytorch network from train()

    Returns:
        action: Single value drawn from legal_moves
    """
    return choose_move_randomly(state, legal_moves)


if __name__ == "__main__":

    debug()
    get_policy()
    ## Example workflow, feel free to edit this! ###
    neural_network = train()
    save_network(neural_network, TEAM_NAME)

    # check_submission(TEAM_NAME)

    neural_network = load_network(TEAM_NAME)

    # Code below plays a single game against a random
    #  opponent, think about how you might want to adapt this to
    #  test the performance of your algorithm.
    def choose_move_no_network(state: np.ndarray,
                               legal_moves: np.ndarray) -> int:
        """The arguments in play_poker() require functions that only take the state as input.

        This converts choose_move() to that format.
        """
        return choose_move(state, legal_moves, neural_network)

    # Challenge your bot to a game of poker!
    play_poker(
        your_choose_move=human_player,
        opponent_choose_move=choose_move_no_network,
        game_speed_multiplier=10,
        render=True,
        verbose=True,
    )
