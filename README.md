# Deep Reinforcement Learning for Spacecraft Active Debris Removal Mission Planning

## Dependencies
To install dependencies, run one of the following command:
* pip command
```bash
pip install -r requirements.txt
```
* conda command
```bash
conda env create -f environment.yml
```

# Run experiment
To train the agent on the ADR env, run the following command:

```bash
python -m src.trainer.trainer
```


# Next steps:
- Exhaustive search
- Evaluate policy / neural network performance
- change model
    - Negative reward if priority debris is not deorbited
    - When a priority is set on a debris, has to be deorbited in x days instead of one removal step (to be modelled in the simulator)