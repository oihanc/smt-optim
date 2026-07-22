import os
import pickle
import numpy as np
from matplotlib import pyplot as plt

import pandas as pd



def order_f0(df: pd.DataFrame, metric_name: str = "budget", ctol: float = 1e-4) -> pd.DataFrame:

    fidelity = df["fidelity"].to_numpy()
    fid_mask = fidelity == np.max(fidelity)

    feas_mask = df["rscv"].to_numpy() <= ctol

    fid_feas_mask = fid_mask & feas_mask

    iter = df["iter"].to_numpy()
    metric = df[metric_name].to_numpy()
    f0 = df["f0"].to_numpy()

    f0_inf = np.where(fid_feas_mask, f0, np.inf)
    f0_ordered = np.minimum.accumulate(f0_inf)

    iter0_mask = iter == 0
    iter0_fmin = np.min(f0_ordered[iter0_mask])

    iter0_max_id = np.where(iter0_mask)[0][-1]

    iter = iter[iter0_max_id:]
    metric = metric[iter0_max_id:]
    f0_ordered = f0_ordered[iter0_max_id:]
    f0_ordered[0] = iter0_fmin

    new_df = pd.DataFrame(data={
        "iter": iter,
        metric_name: metric,
        "fmin": f0_ordered,
    })

    return new_df


def get_convergence_profile(data: pd.DataFrame, metric_name: str = "budget", ctol: float = 1e-4, whitelist: str | None = None) -> dict:

    conv_data = {}


    for algo, a_data in data.items():

        if conv_data.get(algo, None) is None:
            conv_data[algo] = dict()

        for problem, p_data in a_data.items():

            num_inst = len(p_data.keys())

            max_metric = np.empty(num_inst)  # [i_data[-1] for i, i_data in enumerate(a_data.values())]

            all_metric = np.concatenate(
                [i_data[metric_name].to_numpy() for i_data in p_data.values()]
            )

            all_metric = np.unique(all_metric)
            all_metric = np.sort(all_metric)

            # fetch the metric final value for each instance
            for i, i_data in enumerate(p_data.values()):
                max_metric[i] = i_data[metric_name].to_numpy()[-1]

            all_f0 = np.full((all_metric.shape[0], num_inst), np.inf)

            for j, (instance, i_data) in enumerate(p_data.items()):

                ordered_data = order_f0(i_data, metric_name, ctol)

                idx = np.searchsorted(all_metric, ordered_data[metric_name])

                all_f0[idx, j] = ordered_data["fmin"]

                all_f0[:, j] = np.minimum.accumulate(all_f0[:, j], axis=0)

            conv_data[algo][problem] = {
                metric_name: all_metric,
                "max_metric": max_metric,
                "f0": all_f0,
                "mean": np.mean(all_f0, axis=1),
                "std": np.std(all_f0, axis=1),
                "min": np.min(all_f0, axis=1),
                "max": np.max(all_f0, axis=1),
            }

    return conv_data