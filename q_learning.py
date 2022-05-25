import time
import numpy as np
import logging

class Q_learning:
    def __init__(self, util, agent, T=10000, METHOD_NAME="q_learning"):
        self.T = T
        self.METHOD_NAME = METHOD_NAME
        self.util = util
        self.agent = agent
        self.log = logging.getLogger(self.METHOD_NAME)
        logging.basicConfig(level=logging.INFO)
        
    def run(self):
        episode_number = self.util.get_episode_count(self.METHOD_NAME)
        alpha = 0.1
        done = False
        for t in range(episode_number + 1, self.T):
            state = self.agent.reset_game()
            action = self.agent.get_optimal_valid_action(state)
            episode = []
            start = time.time()
            while not done:
                next_state, reward, done, must_remove = self.agent.step(state, action)
                if not done:
                    while next_state == state and must_remove is False:
                        action = (state, np.random.choice(self.agent.unused_entries)["id"])
                        next_state, reward, done, must_remove = self.agent.step(state, action)
                    if must_remove:
                        next_state, reward, done, must_remove = self.agent.step(state, action, must_remove=True)
                        action=(state,next_state)
                        
                    next_action = self.agent.policy(next_state)
            
                if not done:
                    self.agent.q[str(action)] += (alpha*(reward + self.agent.discount * self.agent.q[str(self.agent.policy(next_state))] - self.agent.q[str(action)]))
                else:
                    self.agent.q[str(action)] += (alpha*(reward + (self.agent.discount * 0) - self.agent.q[str(action)]))
                episode.append((state, action, reward))
                action = next_action
                state = next_state
            done=False
            end = time.time()
            # self.agent.board.print_results()
            self.log.info(f"Finished episode {t} in {end-start} seconds")
            self.util.record_results(episode, self.agent.q,end-start,t,self.METHOD_NAME)
            self.agent.board.record_results(t, self.METHOD_NAME)