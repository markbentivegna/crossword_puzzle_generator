import numpy as np
import time
from collections import defaultdict
import logging

class ExploringStarts:
    def __init__(self, util, agent, T=10000, METHOD_NAME="ES"):
        self.T = T
        self.METHOD_NAME=METHOD_NAME
        self.util = util
        self.agent = agent
        self.log = logging.getLogger(self.METHOD_NAME)
        logging.basicConfig(level=logging.INFO)
        
    def generate_episode(self, agent):
        state = agent.reset_game()
        action = agent.policy(state)
        episode_list = []
        while 1:
            next_state, reward, done, must_remove = agent.step(state, action)
            while next_state == state and must_remove is False:
                action = (state, np.random.choice(agent.unused_entries)["id"])
                next_state, reward, done, must_remove = agent.step(state, action)
            episode_list.append((state, action, reward))
            if not done:
                if must_remove:
                    next_state, reward, done, must_remove = agent.step(state, action, must_remove=True)
                    action=(state,next_state)
                next_action = agent.policy(next_state)
            else:
                # agent.board.print_results()
                return episode_list
            state = next_state
            action = next_action

    
    def run(self):
        episode_number = self.util.get_episode_count(self.METHOD_NAME)
        return_sum = defaultdict(float)
        return_count = defaultdict(float)
        for t in range(episode_number + 1, self.T):
            start = time.time()
            new_episode = self.generate_episode(self.agent)
            end = time.time()
            self.log.info(f"Finished episode {t} in {end-start} seconds")
            self.util.record_results(new_episode, self.agent.q,end-start,t,self.METHOD_NAME)
            self.agent.board.record_results(t, self.METHOD_NAME)
            G = 0
            for i in reversed(range(len(new_episode))):
                state, action, reward = new_episode[i]
                prev_sa_pairs = [(step[0], step[1]) for i, step in enumerate(new_episode[:i])]
                G = G*self.agent.discount + reward
                if (state,action) not in prev_sa_pairs:
                    return_sum[(state, action)] += G
                    return_count[(state, action)] += 1
                    self.agent.q[str(action)] = return_sum[(state, action)] / return_count[(state, action)]
                    self.agent.update_policy(state)