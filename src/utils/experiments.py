import os
import importlib
import argparse
import json
from datetime import datetime as dt
from pathlib import Path
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger


def setup(args, hyparams):
    # print all args from get_args in one line
    print("Arguments:")
    print('\n'.join(f'{k}={v}' for k, v in vars(args).items()))

    # print all hyparams from get_hyperparams in one line
    print("Hyparams:")
    print(' '.join(f'{k}={v}' for k, v in hyparams.items()))

    exp_dir = args.experiment_dir + "/" + args.experiment_name + "/" + args.dataset + "/" + str(args.index) + "/" + dt.now().strftime("%Y-%m-%d-%H-%M-%S")
    Path(exp_dir).mkdir(parents=True, exist_ok=True)

    # Save the hyperparameters to a json file
    with open(exp_dir + '/hyperparams.json', 'w') as f:
        json.dump(hyparams, f)
    with open(exp_dir + "/args.json", 'w') as f:
        json.dump(vars(args), f)
    return exp_dir

def augment_config(factor):
    zoom_args = {"height_factor": (0, 0.60/factor), "fill_mode": "constant"}
    blur_args = {"kernel_size": 4, "sigma": 1/factor}
    color_args = {"brightness": 0.4/factor, "jitter": 0.15/factor}
    contrast_args = {"factor": 0.6/factor}
    noise_args = {"intensity": 0.1/factor}
    model_args = {"model_name": "resnet50"}
    config = {"zoom_args": zoom_args, "blur_args": blur_args, "color_args": color_args,
              "contrast_args": contrast_args, "noise_args": noise_args, "model_args": model_args}
    return config

def get_callbacks(exp_dir, monitor):
    return [ModelCheckpoint(filepath=exp_dir + '/best_model_{epoch:02d}', monitor=monitor, 
                            save_best_only=False, save_format='tf', verbose=1, mode='min'),
            CSVLogger(exp_dir + '/results.csv')]

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', required=True, help='Path to the experiment directory')
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--experiment_dir', type=str, default='experiments')
    parser.add_argument('--experiment_name', type=str)
    parser.add_argument('--params', type=str)
    args = parser.parse_args()
    return args

def load_and_run_experiment(experiment_path, args):
    train_script_path = os.path.join(experiment_path, 'train.py')
    
    # Load training module
    spec = importlib.util.spec_from_file_location("train", train_script_path)
    train_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_module)
    
    # Run training with arguments
    train_module.train(args)