
Hypersweeper
==================================================

.. toctree::
   :hidden:
   :maxdepth: 2

   source/getting_started
   source/adding_new_optimizers
   modules
   source/cite

Welcome to the documentation of Hypersweeper, a hydra interface for ask-and-tell hyperparameter optimization.
Hypersweeper uses Hydra to let you parallelize hyperparameter optimization across multiple GPUs and CPUs.
You don't need to worry about compability with your favorite optimizer since it works with any ask-and-tell interface with only minimal coding.
We also include a range of optimizers that you can use out of the box.


How Can You Use the Hypersweeper?
---------------------------------

To use Hypersweeper, you first need to define your target function, i.e. whatever you want to tune, using Hydra as a command line interface.
You should also return the performance value of the target function.
TODO: example
To then do HPO using the Hypersweeper, simply override the default Hydra sweeper with the Hypersweeper variation of your choice.
TODO: example
To parallelize on a cluster, you should additionally override the launcher to whichever one you need.
TODO: example
And of course you also need a search space, for example something like this:
TODO: example
This is the final config for this simple case:
TODO: example
Finally, you can run your optimization using the following command:
TODO: example

Contact
-------

Hypersweeper is developed by `<https://www.automl.org/>`_.
If you want to contribute or found an issue please visit our github page `<https://github.com/automl-private/hypersweeper>`_.
