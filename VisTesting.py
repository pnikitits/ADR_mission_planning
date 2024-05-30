from src.simulator.Simulator import Simulator

if __name__ == "__main__":
    # Setup sim
    simulator = Simulator(starting_index=1, n_debris=10)

    # Test action
    location_frames_df = simulator.strategy_1(action=(9, 29), render=True)

    # Save the df for viewing
    location_frames_df.to_csv("src/vis_data/data_1.csv" , index=True)