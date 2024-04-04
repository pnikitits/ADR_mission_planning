# Deep Reinforcement Learning for Spacecraft Active Debris Removal Mission Planning

## Project structure
```
├── data                                        # Iridium-33 dataset
├── notebooks                                   # Notebooks
│   └── 1.0-exploratory-data-analysis.ipynb     # Notebook for Exploratory data analysis
├── results                                     # Results (exhaustive search results)
├── src                                         # Source folder
│   ├── agent                                   # Agent (replay buffer, action-value network...)
│   ├── config                                  # Config files for all experiments (HPO, exhaustive search...)
│   ├── environment                             # Environment (Space physics equation, transfer strategy...)
│   ├── rlglue                                  # RL-Glue framework (low level protocol to connect agent/env/experiment)
│   ├── simulator                               # Poliastro simulator (to simulate maneuver, propagation...)
│   └── trainer                                 # Trainer function to run the RL experiment
├── exhaustive_search.ipynb                     # Exhaustive search 
├── hyperparams_sweep.py                        # Hyper-Parameter Optimization via Weights and Biases
├── main.py                                     # Main experiment (link the trainer to the config file + Weights and Biases) 
```

## Dependencies
To install dependencies:
* conda command
```bash
conda env create -f environment.yml
```

## Run experiment
To train the agent on the ADR env:

```bash
python main.py
```

## Hyperparameters tuning
To tune the hyperparameters using Weights and Biases:
```bash
python hyperparams_sweep.py -method bayes -nb_sweeps 20
```
By default, the method used is bayesian search and it is configured to execute 200 iterations.
