# Deep Reinforcement Learning for Spacecraft Active Debris Removal Mission Planning

## Dependencies
To install dependencies:
<!-- * conda command
```bash
conda env create -f environment.yml
``` -->

<!-- FYI Environment qui marche: -->
```
python                    3.8.18
pytorch                   2.2.1
astropy                   5.2.2
poliastro                 0.17.0
ipykernel                 6.29.2
tqdm                      4.66.2
wandb                     0.16.3
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
By default, the method used is bayesian search and it is configured to execute 20 iterations.

# Next steps:
- Exhaustive search
- Evaluate policy / neural network performance
- change model
    - Negative reward if priority debris is not deorbited
    - When a priority is set on a debris, has to be deorbited in x days instead of one removal step (to be modelled in the simulator)