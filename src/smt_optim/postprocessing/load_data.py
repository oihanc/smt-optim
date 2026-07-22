import os
import pickle
import numpy as np
from matplotlib import pyplot as plt

import pandas as pd


def load_data(root_dir, ext=".csv", delimiter=","):
    """
    Build a nested dict:
        data[algorithm][problem][instance] -> pandas.DataFrame
    """
    data = {}

    for algorithm in os.listdir(root_dir):
        algo_path = os.path.join(root_dir, algorithm)
        if not os.path.isdir(algo_path):
            continue

        if algorithm.startswith("_"):
                continue

        data[algorithm] = {}

        for problem in os.listdir(algo_path):

            problem_path = os.path.join(algo_path, problem)
            if not os.path.isdir(problem_path):
                continue

            data[algorithm][problem] = {}

            for instance in os.listdir(problem_path):

                instance_path = os.path.join(problem_path, instance)

                for file_name in os.listdir(instance_path):

                    if file_name == "doe.csv":

                        file_path = os.path.join(instance_path, file_name)

                        data[algorithm][problem][instance] = pd.read_csv(file_path) # instance_data

    return data

