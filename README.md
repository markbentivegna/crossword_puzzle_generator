# Crossword Puzzle Generator Using Reinforcement Learning

In this project, I leveraged tabular methods (Q learning and Monte Carlo Exploring Starts) to generate crossword puzzles. This project was completed for a graduate course in Reinforcement Learning. For a more detailed description, see `report.pdf` which includes analysis of techniques, figures, and describes our methodology.

## Instructions

To set up the project, you must first create two directories for crossword puzzles and agent results:

```
    mkdir crossword_puzzles
    mkdir episodes_results
```

Next, unzip the training data CSV file:

```
    unzip train.csv.zip
```

Next, you must execute `main.py` script and pass in the reinforcement learning technique you wish to try. Examples include:

```
    python main.py q_learning
    python main.py es
```

`q_learning` will leverage Q-learning and `es` will use Monte Carlo Exploring Starts algorithm. All other values will run the "random" benchmark that I used for comparison to evaluate the success of the agent.

Please feel free to re-use code for whichever purpose you see fitting.