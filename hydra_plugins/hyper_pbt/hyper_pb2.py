"""PB2: PBT but with a BO backend."""

# A lot of this code is adapted from the original implementation of PB2 Mix/Mult:
# https://github.com/jparkerholder/procgen_autorl
from __future__ import annotations

import logging
from copy import deepcopy

import GPy
import numpy as np
from ConfigSpace.hyperparameters import (NormalIntegerHyperparameter,
                                         UniformIntegerHyperparameter)

from .hyper_pbt import PBT
from .pb2_utils import (TVMixtureViaSumAndProduct, TVSquaredExp, exp3_get_cat,
                        normalize, optimize_acq, standardize, ucb)

log = logging.getLogger(__name__)


class PB2(PBT):
    """PB2: PBT but with a BO backend."""

    def __init__(
        self,
        configspace,
        population_size,
        config_interval,
        seed=42,
        quantiles=None,
        resample_probability=0.25,
        perturbation_factors=None,
        categorical_prob=0.1,
        categorical_fixed=False,
        self_destruct=False,
        categorical_mutation="mix",
        total_iterations=10,  # noqa: ARG002
    ):
        """PB2: PBT but with a BO backend.
        This implemtation covers the BO supported selection of both continuous and categorical hyperparameters.
        """
        super().__init__(
            configspace=configspace,
            population_size=population_size,
            config_interval=config_interval,
            seed=seed,
            quantiles=quantiles,
            resample_probability=resample_probability,
            perturbation_factors=perturbation_factors,
            categorical_prob=categorical_prob,
            categorical_fixed=categorical_fixed,
            self_destruct=self_destruct,
        )
        self.categorical_mutation = categorical_mutation
        self.hierarchical_config = len(self.configspace.get_all_conditional_hyperparameters()) > 0
        self.hp_bounds = np.array(
            [
                [
                    self.configspace.get_hyperparameter(n).lower,
                    self.configspace.get_hyperparameter(n).upper,
                ]
                for n in list(self.configspace.keys())
                if n not in self.categorical_hps
            ]
        )
        self.X = None
        self.model_based = True
        self.total_iterations = 10

    ####################################################################################################

    def get_categoricals(self, config):
        """Get categorical hyperparameter values.

        Parameters
        ----------
        config: Configuration
            A configuration

        Returns:
        -------
        Configuration
            The configuration with new categorical values.
        """
        cats = []
        for i, n in enumerate(self.categorical_hps):
            choices = self.configspace.get_hyperparameter(n).choices
            if self.iteration <= 1 or not all(y > 0 for y in self.ys):
                exp3_ys = self.ys[..., np.newaxis]
            else:
                exp3_ys = normalize(self.ys, self.ys)[..., np.newaxis]
            exp_xs = np.concatenate((self.fixed, self.cat_values, exp3_ys), axis=1)
            cat = exp3_get_cat(choices, exp_xs, self.total_iterations, i + 2)
            config[n] = cat
            cats.append(cat)
        self.cat_current.append(cats)
        return config

    def get_continuous(self, config, performance, X, y):
        """Get continuous hyperparamters from GP.

        Parameters
        ----------
        performance: float
            A performance value
        config: Configuration
            A configuration
        X: List
            Current historical data
        y: List
            Historical performance values

        Returns:
        -------
        Configuration]
            The configuration with new continuous values.
        """
        if len(self.current) == 0:
            m1 = deepcopy(self.m)
            cat_locs = [len(self.X[0]) - x - 1 for x in reversed(range(len(self.cat_values[0])))]
        else:
            # add the current trials to the dataset
            current_use = normalize(self.current, self.hp_bounds.T)
            if self.categorical_mutation == "mix" and len(self.categorical_hps) > 0:
                current_use = np.concatenate((current_use, self.cat_current[:-1]), axis=1)
            padding = np.array(
                [[len(self.config_history) // self.population_size, performance] for _ in range(current_use.shape[0])]
            )
            max_perf = min(*self.performance_history[-self.population_size :], performance)
            padding = normalize(padding, [len(self.config_history) // self.population_size, max_perf])
            current_use = np.hstack((padding, current_use))  # [:, np.newaxis]))
            current_use[current_use <= 0] = 0.01
            Xnew = np.hstack((self.X.T, current_use.T)).T

            # y value doesn't matter, only care about the variance.
            ypad = np.zeros(current_use.shape[0])
            ypad = ypad.reshape(-1, 1)
            if min(y) != max(y):
                y = normalize(y, [min(y), max(y)])
            ynew = np.vstack((y, ypad))
            ynew[ynew <= 0] = 0.01

            if self.categorical_mutation == "mix" and len(self.categorical_hps) > 0:
                cat_locs = [len(self.X[0]) - x - 1 for x in reversed(range(len(self.cat_values[0])))]
                kernel = TVMixtureViaSumAndProduct(
                    self.X.shape[1],
                    variance_1=1.0,
                    variance_2=1.0,
                    variance_mix=1.0,
                    lengthscale=1.0,
                    epsilon_1=0.0,
                    epsilon_2=0.0,
                    mix=0.5,
                    cat_dims=cat_locs,
                )
            else:
                cat_locs = []
                kernel = TVSquaredExp(input_dim=X.shape[1], variance=1.0, lengthscale=1.0, epsilon=0.1)
            Xnew[Xnew >= 0.99] = 0.99  # noqa: PLR2004
            Xnew[Xnew <= 0.01] = 0.01  # noqa: PLR2004
            ynew[ynew >= 0.99] = 0.99  # noqa: PLR2004
            ynew[ynew <= 0.01] = 0.01  # noqa: PLR2004
            m1 = GPy.models.GPRegression(Xnew, ynew, kernel)
            m1.optimize()

        xt = optimize_acq(ucb, self.m, m1, self.fixed, len(self.fixed[0]))
        # convert back...
        if self.categorical_mutation == "mix":
            try:
                cats = [xt[cat_locs]]
            except:  # noqa: E722
                cat_locs = np.array(cat_locs) - self.fixed.shape[1]
                cats = [xt[cat_locs]]
            xt = np.delete(xt, cat_locs)
            if len(xt) > len(self.continuous_hps):
                xt = xt[self.fixed.shape[1] :]
        else:
            cats = self.cat_current

        xt = xt * (np.max(self.hp_bounds.T, axis=0) - np.min(self.hp_bounds.T, axis=0)) + np.min(
            self.hp_bounds.T, axis=0
        )
        xt = xt.astype(np.float32)

        all_hps = [len(self.config_history) // self.population_size, performance]
        xt_ind = 0
        cat_ind = 0
        for n in list(config.keys()):
            if n in self.continuous_hps:
                all_hps.append(xt[xt_ind])
                xt_ind += 1
            else:
                all_hps.append(cats[0][cat_ind])
                cat_ind += 1

        curr = []
        for v, n in zip(xt, self.continuous_hps, strict=False):
            hp = self.configspace.get_hyperparameter(n)
            value = int(v) if isinstance(hp, NormalIntegerHyperparameter | UniformIntegerHyperparameter) else float(v)
            config[n] = max(hp.lower, min(value, hp.upper))
            curr.append(max(hp.lower, min(value, hp.upper)))
        self.current.append(curr)
        return config

    def perturb_hps(self, config, performance, is_good):
        """Suggest next configuration.

        Parameters
        ----------
        performance: List[float]
            A list of the latest agent performances
        config: List[Configuration]
            A list of the recent configs
        is_good: bool
            does this config belong to the best quantile

        Returns:
        -------
        Configuration
            The next configuration.
        """
        if is_good:
            return config
        if self.categorical_mutation == "mult":
            for i, n in enumerate(self.categorical_hps):
                config[n] = self.cat_current[0][i]
        else:
            config = self.get_categoricals(config)
        if len(self.continuous_hps) > 0:
            config = self.get_continuous(config, performance, self.X, self.y)
        return config

    def get_model_data(self, performances=None, configs=None):
        """Parse history for relevant data.

        Parameters
        ----------
        performances: List[float]
            A list of the latest agent performances
        configs: List[Configuration]
            A list of the recent configs
        """
        if self.categorical_mutation == "mult":
            # We get the categoricals based on all performance data, then filter everything after
            ys = []
            tps = []
            self.cat_values = []
            for i in reversed(range(1000 // self.population_size)):
                for j in range(self.population_size):
                    t = len(self.performance_history) // self.population_size - i
                    if t - i <= i:
                        continue
                    p = self.performance_history[-i * j]
                    ys.append(self.performance_history[-(i - 1) * j] - self.performance_history[-i * j])
                    tps.append([t, p])
                    config = self.config_history[-j * i]
                    cat = [
                        v
                        for v, n in zip(list(config.values()), list(config.keys()), strict=False)
                        if n in self.categorical_hps
                    ]
                    self.cat_values.append(cat)
            self.ys = np.array(ys)
            self.fixed = normalize(tps, [len(self.performance_history // self.population_size), max(performances)])
            self.cat_current = []
            self.get_categoricals(configs[0])
            # Now filter data to user for the continuous variables
            performance_data, config_data = deepcopy(self.performance_history), deepcopy(self.config_history)
            iterations_run = len(config_data) // self.population_size
            to_keep = []
            for i in range(self.population_size):
                for j in range(iterations_run):
                    cvs = [
                        v
                        for v, n in zip(config_data[i * j].values(), config_data[i * j].keys(), strict=False)
                        if n in self.categorical_hps
                    ]
                    if all(old == new for old, new in zip(cvs, self.cat_current[0], strict=False)):
                        to_keep.append(j * i)
            performance_data = [performance_data[i] for i in to_keep]
            config_data = [config_data[i] for i in to_keep]
        else:
            performance_data, config_data = self.performance_history, self.config_history

        all_hps = []
        hp_values = []
        self.cat_values = []
        ts = []
        ps = []
        ys = []
        for i in reversed(range(1000 // self.population_size)):
            for j in range(self.population_size):
                t = len(performance_data) // self.population_size - i
                if t <= 0:
                    continue
                ts.append(t)
                p = performance_data[-i * j]
                ys.append(performance_data[-(i - 1) * j] - p)
                config = config_data[-j * i]
                hps = [
                    v
                    for v, n in zip(list(config.values()), list(config.keys()), strict=False)
                    if n in self.continuous_hps
                ]
                cat = [
                    v
                    for v, n in zip(list(config.values()), list(config.keys()), strict=False)
                    if n in self.categorical_hps
                ]
                all_hp = list(config.values())
                all_hps.append(all_hp)
                self.cat_values.append(cat)
                hp_values.append(hps)
                ps.append(p)

        # current_best_values = list(current_best[-1].values())
        self.ts = np.array(ts)
        self.hp_values = np.array(hp_values)
        self.all_hps = np.array(all_hps)
        self.ys = np.array(ys)

        if len(self.continuous_hps) > 0:
            self.X = normalize(self.hp_values, self.hp_bounds.T)
        else:
            self.X = self.hp_values
        self.y = standardize(self.ys).reshape(self.ys.size, 1)
        # If all values are the same, don't normalize to avoid nans, instead just cap.
        # This probably only happens if improvement is 0 for all anyway.
        if min(self.y) != max(self.y):
            self.y = normalize(self.y, [min(self.y), max(self.y)])

        max_perf = min(*self.performance_history[-self.population_size :], *performances)
        min_perf = max(*self.performance_history[-self.population_size :], *performances)
        self.ts = normalize(self.ts, [0, len(config_data) // self.population_size])
        ps = normalize(ps, [min_perf, max_perf])
        self.fixed = np.array([[t, p] for t, p in zip(self.ts, ps, strict=False)])
        if self.X is not None:
            self.X = np.concatenate((self.fixed, self.X), axis=1)
        else:
            self.X = self.fixed

        self.X[self.X <= 0] = 0.01
        self.X[self.X >= 1] = 0.99

    def fit_model(self, performances, configs):
        """Fit the GP with current data.

        Parameters
        ----------
        performances: List[float]
            A list of the latest agent performances
        configs: List[Configuration]
            A list of the recent configs
        """
        self.get_model_data(performances, configs)

        if self.categorical_mutation == "mix" and len(self.categorical_hps):
            self.X = np.concatenate((self.X, self.cat_values), axis=1)
            # self.fixed = np.concatenate((self.fixed, self.cat_values), axis=1)
            cat_locs = [len(self.X[0]) - x - 1 for x in reversed(range(len(self.cat_values[0])))]

            kernel = TVMixtureViaSumAndProduct(
                self.X.shape[1],
                variance_1=1.0,
                variance_2=1.0,
                variance_mix=1.0,
                lengthscale=1.0,
                epsilon_1=0.0,
                epsilon_2=0.0,
                mix=0.5,
                cat_dims=cat_locs,
            )
        else:
            kernel = TVSquaredExp(input_dim=self.X.shape[1], variance=1.0, lengthscale=1.0, epsilon=0.1)

        self.X = np.nan_to_num(self.X)
        self.y = np.nan_to_num(self.y)
        self.X[self.X <= 0.01] = 0.01  # noqa: PLR2004
        self.X[self.X >= 0.99] = 0.99  # noqa: PLR2004
        self.y[self.y <= 0.01] = 0.001  # noqa: PLR2004
        self.y[self.y >= 0.99] = 0.99  # noqa: PLR2004

        try:
            self.m = GPy.models.GPRegression(self.X, self.y, kernel)
            self.m.optimize()
        except np.linalg.LinAlgError:
            # add diagonal ** we would ideally make this something more robust...
            self.X += np.ones(self.X.shape) * 1e-3
            self.X[self.X <= 0.01] = 0.01  # noqa: PLR2004
            self.X[self.X >= 0.99] = 0.99  # noqa: PLR2004
            self.y[self.y <= 0.01] = 0.01  # noqa: PLR2004
            self.y[self.y >= 0.99] = 0.99  # noqa: PLR2004
            self.m = GPy.models.GPRegression(self.X, self.y, kernel)
            self.m.optimize()

        self.m.kern.lengthscale.fix(self.m.kern.lengthscale.clip(1e-5, 1))
        self.current = []
        if self.categorical_mutation != "mult":
            self.cat_current = []


def make_pb2(configspace, pbt_args):
    """Make a PBT instance for optimization."""
    return PB2(configspace, **pbt_args)
