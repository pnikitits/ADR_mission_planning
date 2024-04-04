# Deep Reinforcement Learning for Spacecraft Active Debris Removal Mission Planning

## Project structure
```
├── data                                        # a
├── notebooks                                   # b
│   └── 1.0-exploratory-data-analysis.ipynb     # c
├── results                                     # d
│   └── python                                  # e
├── src                                         # f
│   ├── agent                                   # g
│   ├── config                                  # g
│   ├── environment                             # g
│   ├── rlglue                                  # g
│   ├── config                                  # g
│   ├── simulator                               # g
│   └── trainer                                 # g
├── exhaustive_search.ipynb                     # g
├── hyperparams_sweep.py                        # g
├── main.py                                     # g



```
+---data
+---notebooks
+---results
+---src
    +---agent
    +---config
    +---environment
    +---rlglue
    +---simulator
    +---trainer


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
