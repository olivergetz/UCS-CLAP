import numpy as np
import datetime
import json
import os

class TrainLogger():
    def __init__(self, log_directory : str, settings):
        """
        Logs info about the training session to a json file. 

        Args:
        - log_directory: where to store the logs.
        - settings: data from a json file that hold hyper-parameter values and layer info.
        """

        self.log_directory = log_directory
        self.log_start_time = datetime.datetime.now().strftime('%Y_%m_%d-%H_%M_%S')
        self.model_name = settings["model_name"]
        self.dataset = settings["dataset_path"]
        self.dataset_name = self.dataset.split("/")[-1]         # Last segment of the path should be used as the dataset name.
        self.dataset_name = self.dataset_name.split(".")[0]     # Remove extension

        self.log = {
            "time": self.log_start_time,
            "parameters": 0,
            "trainable_parameters": 0,
            "settings": settings,
            "epochs": {}
        }

    def log_epoch(self, data:dict, epoch:int):
        """
        Args:
            - data: Dictionary with epoch-specific data.
            - epoch: the current epoch. Add 1 if 0-indexed epochs are not desired.
        """
        self.log["epochs"][epoch] = data

    def save_log(self):
        with open(os.path.join(self.log_directory, f"{self.model_name}_{self.dataset_name}_{self.log_start_time}.json"), "w") as f:
            f.write(json.dumps(self.log, cls=NpEncoder))


class NpEncoder(json.JSONEncoder):
    """Source: https://stackoverflow.com/questions/50916422/python-typeerror-object-of-type-int64-is-not-json-serializable"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
    
if __name__ == "__main__":

    with open ('training/model_definitions/vae.json', 'r') as f:
        settings = json.load (f)

    logger = TrainLogger("training/generative_experiments/audio_results", settings)