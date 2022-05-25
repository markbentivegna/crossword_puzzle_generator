import pandas as pd
from agent import Agent
from game_board import GameBoard
from util import Utility
from q_learning import Q_learning
from exploring_starts import ExploringStarts
from baseline import Baseline
import argparse

parser = argparse.ArgumentParser(description="Run crossword puzzle generators")
parser.add_argument("method", metavar="method", type=str, help="Reinforcement learning technique to use when generating puzzles")
args = parser.parse_args()
method = args.method

train_df = pd.read_csv("train.csv")
train_df["answer"].drop_duplicates(inplace=True)
train_df["answer"] = train_df["answer"].str.lower()

word_list = []
for word in train_df["answer"].tolist():
  if "x" in word or "q" in word or "z" in word:
    pass
  else:
    word_list.append(word)

board = GameBoard(word_list,input_file="medium.txt")
agent = Agent(board)
agent.get_help()
util = Utility(agent)

if method == "q_learning":
  Q_learning(util, agent).run()
elif method == "es":
  ExploringStarts(util, agent).run()
else:
  Baseline(util, agent).random_simulations()
