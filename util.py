import os
import re
import json

class Utility:
    def __init__(self, agent):
        self.agent = agent
    
    def get_episode_count(self, method):
        episode_numbers = []
        for filename in os.listdir("episode_results/"):
            if filename.endswith(f"_{method}.json"):
                episode_numbers.append(int(re.findall(r'\d+', filename)[0]))
        if len(episode_numbers) > 0:
            return max(episode_numbers)
        return 0

    def get_q_values(self, filename):
        with open(filename, "r") as f:
            self.agent.q = json.load(f)["q_values"]
    
    def record_results(self, episode_list,q_values, runtime, episode_number,method):
        episode_results = {
            "episode_number": episode_number,
            "runtime": runtime,
            "q_values": dict(q_values),
            "episode_list": episode_list
        }
        with open(f"episode_results/episode{episode_number}_{method}.json", 'w') as f:
            json.dump(episode_results, f)