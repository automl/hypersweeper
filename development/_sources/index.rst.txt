
Hypersweeper
==================================================

.. toctree::
   :hidden:
   :maxdepth: 2

   source/getting_started
   source/adding_new_optimizers
   api
   source/cite

Welcome to the documentation of Hypersweeper, a hydra interface for ask-and-tell hyperparameter optimization.
Hypersweeper uses Hydra to let you parallelize hyperparameter optimization across multiple GPUs and CPUs.
You don't need to worry about compability with your favorite optimizer since it works with any ask-and-tell interface with only minimal coding.
We also include a range of optimizers that you can use out of the box.


How Can You Use the Hypersweeper?
---------------------------------

To use Hypersweeper, you first need to define your target function, i.e. whatever you want to tune, using Hydra as a command line interface.
You should also return the performance value of the target function.

.. code-block:: python
  :linenos:

  import hydra

  @hydra.main(config_path="configs", config_name="example", version_base="1.1")
   def my_target_function(cfg):
      performance = cfg.x - cfg.y
      return performance

   if __name__ == "__main__":
      branin()

To then do HPO using the Hypersweeper, simply override the default Hydra sweeper with the Hypersweeper variation of your choice. You should also include a search space here:

.. code-block:: python

  defaults:
  - _self_
  - override hydra/sweeper: HyperRS

   x: 3
   y: 4


   hydra:
   sweeper:
      n_trials: 10
      search_space:
         hyperparameters:
         x:
            type: uniform_float
            lower: -5
            upper: 15
         y:
            type: uniform_float
            lower: 2
            upper: 15

   

To parallelize on a cluster, you should additionally override the launcher to whichever one you need.

.. code-block:: python

  defaults:
  - _self_
  - override hydra/sweeper: HyperRS
  - override hydra/launcher: submitit_slurm
  ...

Finally, you can run your optimization using the following command:

.. code-block:: bash

   python my_target_function.py -m

For more examples, please refer to the `examples` folder in our repository: `<https://github.com/automl-private/hypersweeper/tree/main/examples>`_.

Contact
-------

Hypersweeper is developed by `<https://www.automl.org/>`_.
If you want to contribute or found an issue please visit our github page `<https://github.com/automl-private/hypersweeper>`_.
