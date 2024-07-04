from src.utils.experiments import get_args, load_and_run_experiment
from datetime import datetime as dt
"""
Runs an experiment by loading the train.py script from the finetuning directory and running the train function.
This is primarily used for the count decoding phase to extract true object-quantity from
pre-trained feature encoders.
"""
if __name__ == '__main__':
    args = get_args()
    experiment_path = args.experiment
    print("Running experiment at: " + experiment_path)
    print("Current time: " + dt.now().strftime("%Y-%m-%d-%H-%M-%S"))
    load_and_run_experiment(experiment_path, args)