import numpy as np
import time
import json
import logging

class Baseline:
    def __init__(self, util, agent, T=1000, METHOD_NAME="random"):
        self.T = T
        self.METHOD_NAME = METHOD_NAME
        self.util = util
        self.agent = agent
        self.log = logging.getLogger(self.METHOD_NAME)
        logging.basicConfig(level=logging.INFO)

    def record_random_results(self, episode_count, runtime, episode_number):
        episode_results = {
            "episode_number": episode_number,
            "runtime": runtime,
            "episode_count": episode_count
        }
        with open(f"episode_results/episode{episode_number}_{self.METHOD_NAME}.json", 'w') as f:
            json.dump(episode_results, f)

    def random_simulations(self):
        episode_number = self.util.get_episode_count(self.METHOD_NAME)
        for t in range(episode_number + 1, self.T):
            episode_length = 0
            self.agent.reset_game()
            start = time.time()
            while not self.agent.is_puzzle_complete():
                available_entries = self.agent.get_available_entries()
                while len(available_entries) == 0:
                    random_entry = np.random.choice(self.agent.used_entries)
                    self.agent.remove_word(random_entry)
                    episode_length += 1
                    available_entries = self.agent.get_available_entries()
                random_entry = np.random.choice(available_entries)
                valid_words_list = self.agent.get_valid_words_for_entry(random_entry)
                random_word = np.random.choice(valid_words_list)
                episode_length += 1
                self.agent.insert_word(random_entry, random_word)
            end = time.time()
            self.log.info(f"Finished episode {t} in {end-start} seconds")
            self.record_random_results(episode_length, end-start, t)
            self.agent.board.record_results(t, self.METHOD_NAME)