from src.simulator.Simulator import Simulator
import pandas as pd

if __name__ == "__main__":
    # Parameters
    step_sec = 15 # 10 seconds per step?
    sequence = [1,2,3]
    starting_index = 0

    # Setup sim
    simulator = Simulator(starting_index=starting_index , n_debris=5)

    # Test action
    all_frames = pd.DataFrame()
    for idx in sequence:
        transfer_frames = simulator.strategy_1(action=(idx, 29), render=True, step_sec=step_sec)
        all_frames = pd.concat([all_frames, transfer_frames], axis=0)

    # Save the df for viewing
    all_frames.to_csv("src/vis_data/data_1.csv" , index=True)