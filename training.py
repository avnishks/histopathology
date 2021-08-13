import copy
import logging
import random
import warnings
from collections import OrderedDict

import numpy as np
from oi_core import resolve
from oi_core.create import create
from oi_core.resolve import resolve_all_configs


import ai.histopath_template.definitions.manager.mlflow.logger as mlflow_logger


import tensorflow as tf


_LOG = logging.getLogger(__name__)

@mlflow_logger.log_experiment(nested=True)
@mlflow_logger.log_metric('best_metric')
def train(cfg):

    cfg = resolve_all_configs(cfg)
    print("Our config:", cfg)
    seed = cfg['seed']
    num_epoch = cfg['epoch']

    # Setting the seed.
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)

    # Set GPU memory growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        # Dataset
        train_generator = create(cfg["dataset"]["train"])
        valid_generator = create(cfg["dataset"]["valid"])
        # Model
        model = create(cfg["model"])
        model.summary()

        # Optimizer
        optimizer = create(cfg["optimizer"])

        # Callback
        callbacks = [create(_cfg) for _cfg in cfg["callbacks"]]

        # Compilation
        model.compile(loss='categorical_crossentropy',
                    optimizer=optimizer,
                    metrics=['accuracy'])

        # Training
        history = model.fit(train_generator,
                            epochs=num_epoch,
                            validation_data=valid_generator,
                            callbacks=callbacks)
        #history = model.predict(valid_generator)

    best_metric = np.max(np.array(history.history['val_accuracy']))

    tf.keras.backend.clear_session()
    return best_metric
    
@mlflow_logger.log_experiment(nested=False)
def train_skopt(cfg, n_iter, base_estimator, n_initial_points, random_state, train_function=train):

    """
    Do a Bayesian hyperparameter optimization.

    :param cfg: Configuration file.
    :param n_iter: Number of Bayesien optimization steps.
    :param base_estimator: skopt Optimization procedure.
    :param n_initial_points: Number of random search before starting the optimization.
    :param random_state: seed.
    :param train_function: The trainig procedure to optimize. The function should take a dict as input and return a metric maximize.
    :return:
    """

    import skopt
    import yaml
    from skopt.space import Categorical, Integer, Real

    def recursive_update(to_update, config):

        from skopt.space import Categorical, Integer, Real

        new_config = copy.deepcopy(config)
        for key, value in new_config.items():

            if isinstance(value, dict):
                new_config[key] = recursive_update(to_update, value)

            if isinstance(value, str) and '!skopt' in value:
                this_op = value.split('!skopt')[-1]

                try:
                    update_key = eval(this_op)._name
                except:
                    update_key = this_op

                new_config[key] = to_update[update_key]

        return new_config

    def recursive_extract(config):

        skopt_ops = {}

        def inner_extract(sub_config):

            new_config = copy.deepcopy(config)
            for key, value in sub_config.items():

                if isinstance(value, dict):
                    inner_extract(value)

                if isinstance(value, str) and '!skopt' in value:
                    this_op = value.split("!skopt")[-1]
                    try:
                        x = eval(this_op)
                        if isinstance(x, (Real, Integer, Categorical)):
                            skopt_ops[x._name] = x

                    except:
                        pass

        inner_extract(config)
        return skopt_ops

    # Extract the space
    shared_skopt = recursive_extract(cfg)

    # Create the optimizer
    optimizer = skopt.Optimizer(dimensions=shared_skopt.values(),
                                base_estimator=base_estimator,
                                n_initial_points=n_initial_points,
                                random_state=random_state)

    # Do a bunch of loops.
    for _ in range(n_iter):

        suggestion = optimizer.ask()
        suggestion_tmp = {k: v for k, v in zip(shared_skopt.keys(), suggestion)}
        this_cfg = recursive_update(suggestion_tmp, cfg)

        try:
            optimizer.tell(suggestion, - train_function(this_cfg)) # We minimize the negative accuracy/AUC
        except RuntimeError as e:
            print("The following error was raised:\n {} \n, launching next experiment.".format(e))
            optimizer.tell(suggestion, 0.)  # Something went wrong, (probably a CUDA error).

    # Done! Hyperparameters tuning has never been this easy.
