import numpy as np
import copy
import random
from collections import defaultdict

class Agent():
    def __init__(self, board,gamma=0.9,epsilon=0.1,discount=0.9):
        self.board = board
        self.gamma = gamma
        self.epsilon = epsilon
        self.discount = discount
        self.policies = defaultdict(str)
        self.action_space = [0,1]
        self.state_space = defaultdict(list)
        self.rewards = []
        self.q = defaultdict(float)
        self.word_combinations = {}
        self.reset_game()
        self.random_initialize_q()
        
    def random_initialize_q(self):
        word_ids = [subset["id"] for subset in self.unused_entries]
        for id in word_ids:
            for other_id in word_ids:
                if other_id is not id:
                    self.q[str((id, other_id))] = np.random.uniform()
                    
    def update_policy(self, state):
        max_q = -999999
        word_ids = [subset["id"] for subset in self.unused_entries]
        for id in word_ids:
            if self.q[str((state,id))] > max_q:
                self.policies[state] = (state,id)
                max_q = self.q[str((state,id))]
                
    def is_word_id_used(self, word_id):
        return word_id in [entry["id"] for entry in self.used_entries]
        
    def validate_action(self, action, state):
        from_state = action[0]
        to_state = action[1]
        if from_state in [entry["id"] for entry in self.used_entries] and to_state in [entry["id"] for entry in self.unused_entries]:
            return True
        return False
        
    def random_action(self,id):
        return (id, self.get_random_unused_entry()["id"])
        
    def get_expected_q_greedy(self, state):
        word_ids = [subset["id"] for subset in self.unused_entries]
        temp_q = 0
        for id in word_ids:
            if self.get_optimal_valid_action(state)[1] == id:
                temp_q += ((1-self.epsilon)* self.q[str((state,id))])
            else:
                temp_q += (self.epsilon) * self.q[str((state,id))]
        return temp_q
    
    def get_optimal_valid_action(self, state, update_policy=True, is_random=False):
        word_ids = [subset["id"] for subset in self.unused_entries]
        if is_random:
            if np.random.uniform() < self.epsilon:
                return self.random_action(state)
        max_q = -999999
        optimal_action = self.random_action(state)
        for id in word_ids:
            if self.q[str((state,id))] > max_q and self.validate_action((state,id),state):
                optimal_action = (state,id)
                max_q = self.q[str((state,id))]
        if update_policy:
            self.policies[state] = optimal_action
        return optimal_action
        
    def get_optimal_action(self, state):
        word_ids = [subset["id"] for subset in self.unused_entries]
        max_q = -999999
        optimal_action = self.random_action(state)
        for id in word_ids:
            if self.q[str((state,id))] > max_q:
                optimal_action = (state,id)
                max_q = self.q[str(optimal_action)]
        return optimal_action
        
    def policy(self, state):
        if state not in self.policies:
            self.policies[state] = self.get_optimal_valid_action(state)
        if not self.validate_action(self.policies[state], state):
            return self.get_optimal_valid_action(state)
        return self.policies[state]
        
    def initialize_word_entries(self):
        for word_entry in self.board.state_space["word_entries"]:
            self.unused_entries.append(word_entry)
            self.state_space["unused"].append(word_entry["id"])
            
    def get_words_of_size(self, size):
        return [word for word in self.board.word_list if len(word) == size]
        
    def insert_word(self, word_entry, word):
        start=word_entry["start"]
        end=word_entry["end"]
        direction = word_entry["id"][-1]
        new_grid = copy.deepcopy(self.board.grid)
        if direction == "D":
            for i in range(start[0],end[0] + 1):        
                new_grid[i][start[1]] = word[i-start[0]]
        elif direction == "A":
            for i in range(start[1],end[1] + 1):
                new_grid[start[0]][i] = word[i-start[1]]
                
        self.used_entries.append(word_entry)
        self.state_space["used"].append(word_entry["id"])
        self.unused_entries.remove(word_entry)
        self.state_space["unused"].remove(word_entry["id"])
        self.board.grid = new_grid
    
    def get_random_word(self):
        return np.random.choice(self.board.word_list)
    
    def get_random_word_of_size(self, size):
        valid_words = self.get_words_of_size(size)
        return np.random.choice(valid_words)
        
    def is_word_entry_available(self, word_entry):
        start = word_entry["start"]
        end = word_entry["end"]
        distance = word_entry["distance"]
        direction = word_entry["id"][-1]
        occupied_grid_count = 0
        if direction == "D":
            for i in range(start[0],end[0] + 1):
                if self.board.grid[i][start[1]] != self.board.delim:
                    occupied_grid_count += 1
        elif direction == "A":
            for i in range(start[1],end[1] + 1):
                if self.board.grid[start[0]][i] != self.board.delim:
                    occupied_grid_count += 1
        if occupied_grid_count == distance:
            return False            
        return True

    def extract_word(self, direction, grid,start,end):
        word = ""
        if direction == "D":
            for i in range(start[0],end[0] + 1):
                word += grid[i][start[1]]
        elif direction == "A":
            for i in range(start[1],end[1] + 1):
                word += grid[start[0]][i]
        return word
    
    def validate_new_grid(self, new_grid):
        for word_entry in self.board.state_space["word_entries"]:
            if not self.is_word_entry_available(word_entry):
                direction = word_entry["id"][-1]
                start = word_entry["start"]
                end = word_entry["end"]
                word = self.extract_word(direction, new_grid,start,end)
        if word != "" and word is not None and word not in self.board.word_list:
            return False
        return True
        
    def get_random_word_entry(self):
        return np.random.choice(self.board.state_space["word_entries"])
        
    def get_random_used_entry(self):
        return np.random.choice(self.used_entries)
        
    def get_random_unused_entry(self):
        return np.random.choice(self.unused_entries)
        
    def reset_game(self):
        self.board.grid = self.board.initialize_grid()
        self.cumulative_reward = 0
        self.used_entries = []
        self.unused_entries = []
        self.initialize_word_entries()
        
        word_entry = self.get_entry_by_id("2D")
        valid_word_list = self.get_valid_words_for_entry(word_entry)
        random_word = np.random.choice(valid_word_list)
        self.insert_word(word_entry, random_word)
        return "2D"
    
    def used_entries_containing_square(self, coordinates):
        used_entries_containing_square = []
        entries_containing_square = self.entries_containing_square(coordinates)
        for temp_entry in entries_containing_square:
            if temp_entry in self.used_entries and temp_entry not in used_entries_containing_square:
                used_entries_containing_square.append(temp_entry)
        return used_entries_containing_square
        
    def entries_containing_square(self, coordinates):
        entries_list = []
        for word_entry in self.board.state_space["word_entries"]:
            direction = word_entry["id"][-1]
            start = word_entry["start"]
            end = word_entry["end"]
            if direction == "D":
                for i in range(start[0],end[0] + 1):
                    if [i,start[1]] == coordinates:
                        entries_list.append(word_entry)
            elif direction == "A":
                for i in range(start[1],end[1] + 1):
                    if [start[0],i] == coordinates:
                        entries_list.append(word_entry)
        return entries_list
        
    def get_neighboring_used_entries(self, word_entry):
        direction = word_entry["id"][-1]
        start = word_entry["start"]
        end = word_entry["end"]
        
        neighboring_used_entries = []
        
        if direction == "D":
            for i in range(start[0],end[0] + 1):
                used_entries_containing_square = self.used_entries_containing_square([i,start[1]])
                for entry in used_entries_containing_square:
                    if entry not in neighboring_used_entries:
                        neighboring_used_entries.append(entry)
        elif direction == "A":
            for i in range(start[1],end[1] + 1):
                used_entries_containing_square = self.used_entries_containing_square([start[0],i])
                for entry in used_entries_containing_square:
                    if entry not in neighboring_used_entries:
                        neighboring_used_entries.append(entry)                
        return neighboring_used_entries
        
    def remove_word(self, word_entry):
        direction = word_entry["id"][-1]
        start = word_entry["start"]
        end = word_entry["end"]
        self.used_entries.remove(word_entry)
        self.state_space["used"].remove(word_entry["id"])
        self.unused_entries.append(word_entry)
        self.state_space["unused"].append(word_entry["id"])
        if direction == "D":
            for i in range(start[0],end[0] + 1):
                used_entries_containing_square = self.used_entries_containing_square([i,start[1]])
                if len(used_entries_containing_square) == 0:
                    self.board.grid[i][start[1]] = self.board.delim
        elif direction == "A":
            for i in range(start[1],end[1] + 1):
                used_entries_containing_square = self.used_entries_containing_square([start[0],i])
                if len(used_entries_containing_square) == 0:
                    self.board.grid[start[0]][i] = self.board.delim
                    
    def compare_words(self, constraint_chars, word):
        for i in range(len(constraint_chars)):
            if constraint_chars[i] != self.board.delim:
                if constraint_chars[i] != word[i]:
                    return False
        return True
        
    def get_valid_words(self, word_entry):
        valid_words = []
        direction = word_entry["id"][-1]
        start = word_entry["start"]
        end = word_entry["end"]
        distance = word_entry["distance"]
        constraint_chars = self.extract_word(direction, self.board.grid,start,end)
        potential_matches = self.get_words_of_size(distance)
        random.shuffle(potential_matches)
        for word in potential_matches:
            if len(valid_words) > 100:
                return valid_words
            if self.compare_words(constraint_chars, word):
                valid_words.append(word)
        return valid_words
    
    def get_valid_words_for_entry(self,word_entry):
        return self.get_valid_words(word_entry)
    
    def get_unused_word_entries(self):
        return self.unused_entries
        
    def get_used_word_entries(self):
        return self.used_entries
        
    def get_entry_from_list(self, entries, id):
        for entry in entries:
            if entry["id"] == id:
                return entry
    
    def get_entry_by_id(self, id):
        used_ids = [entry["id"] for entry in self.used_entries]
        if id in used_ids:
            return self.get_entry_from_list(self.used_entries, id)
        else:
            return self.get_entry_from_list(self.unused_entries, id)
            
    def get_available_entries(self):
        available_entries = []
        for entry in self.get_unused_word_entries():
            valid_word_list = self.get_valid_words_for_entry(entry)
            if len(valid_word_list) > 0:
                available_entries.append(entry)
        return available_entries
        
    def step(self, state, action_tuple, must_remove=False):
        action = int(must_remove==True)
        next_state = action_tuple[1]
        if next_state not in [entry["id"] for entry in self.unused_entries]:
            return self.step(state, (state,self.get_random_unused_entry()["id"]))
        word_entry = self.get_entry_by_id(next_state)
        if action == 0:
            valid_word_list = self.get_valid_words_for_entry(word_entry)
            if len(valid_word_list) == 0:
                if len(self.get_available_entries()) == 0:
                    return (state, -1, False, True)
                else:
                    return (state, -1, False, False)
            random_word = np.random.choice(valid_word_list)
            self.insert_word(word_entry, random_word)
            reward = self.get_reward()
            if reward == 0:
                done = True
            else:
                done = False
            return (next_state, reward, done, False)
        if action == 1:
            fail_count = 0
            valid_word_list = []
            while len(valid_word_list) == 0:
                fail_count += 1
                while len(self.get_available_entries()) == 0 and len(self.get_unused_word_entries()) > 0:
                    random_unused_entry = np.random.choice(self.get_unused_word_entries())
                    random_entry = np.random.choice(self.get_neighboring_used_entries(random_unused_entry))
                    self.remove_word(random_entry)
                    potential_entries = copy.copy(self.get_unused_word_entries())
                random_entry = np.random.choice(potential_entries)
                valid_word_list = self.get_valid_words_for_entry(random_entry)
            random_word = np.random.choice(valid_word_list)
            self.insert_word(random_entry, random_word)
            return (random_entry["id"], -fail_count, False, False)
            
    def is_puzzle_complete(self):
        return len(self.unused_entries) == 0
        
    def get_reward(self):
        if self.is_puzzle_complete():
            return 0
        return -1
        
    def get_help(self):
        self.q[str(("2D","3D"))] = 1000
        self.update_policy("2D")
        self.q[str(("3D", "11D"))] = 1000
        self.update_policy("3D")
        self.q[str(("11D", "15D"))] = 1000
        self.update_policy("11D")
