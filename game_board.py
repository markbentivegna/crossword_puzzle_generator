from collections import defaultdict
import numpy as np

class GameBoard():
    def __init__(self, word_list, n=13, input_file=None, delim=" ", blank_space="+"):
        self.height = n
        self.width = n
        self.word_list = word_list
        self.delim = delim
        self.blank_space = blank_space
        self.state_space = {}
        self.state_space = self.load_input_file(input_file)
        self.grid = self.initialize_grid()
        
    def load_input_file(self, filename):
        i, j = 0, 0
        state_space = defaultdict(list)
        with open(filename, "r") as f:
            for line in f.readlines():
                j = 0
                row = line.split(' ')
                for char in row:
                    try:
                        if int(char) == 0:
                            state_space["grid"].append((i,j))
                    except:
                        word_entry = row[0].rstrip()
                        id = word_entry.split(":")[0]
                        [int(s) for s in word_entry.split(":")[1].split() if s.isdigit()]
                        start = np.array([int(word_entry.split(":")[1].split(",")[0]), int(word_entry.split(":")[1].split(",")[1])])
                        end = np.array([int(word_entry.split(":")[2].split(",")[0]), int(word_entry.split(":")[2].split(",")[1])])
                        state_space["word_entries"].append({
                            "id": id,
                            "start": start,
                            "end": end,
                            "distance": np.linalg.norm((start-end)) + 1
                        })
                    j += 1
                i += 1
        return state_space
      
    def print_results(self):
        for i in range(self.height):
            row_str = ""
            for j in range(self.width):
                row_str += self.grid[i][j]
                row_str += " "
            print(row_str)
            
    def record_results(self, episode_number, method):
        row_list = []
        for i in range(self.height):
            row_str = ""
            for j in range(self.width):
                row_str += self.grid[i][j]
                row_str += " "
            row_list.append(f"{row_str}\n")
        with open(f"crossword_puzzles/episode{episode_number}_{method}.txt", 'w') as f:
            f.writelines(row_list)
            
    def initialize_grid(self):
        grid = []
        for i in range(self.height):
            current_row = []
            for j in range(self.width):
                if (i, j) in self.state_space["grid"]:
                    current_row.append(self.delim)
                else:
                    current_row.append(self.blank_space)
            grid.append(current_row)
        return grid