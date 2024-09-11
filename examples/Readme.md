# Hypersweeper Examples

To try out the different hypersweepers, we provide three simple examples:
- the oldie but goldie: optimizing Branin
- the AutoML basic: configuring a small MLP
- the RL excursion: finding good hyperparameters for a SAC agent

The base scripts for these can be found in the corresponsing python files - you'll see they only contain the function evaluation, but have a config "cfg" which contains the values to be configured. Using hydra, we can use the hypersweeper to get these values from different optimizers.

## Navigating the Configuration Files

Hypersweeper is based on [Hydra](https://hydra.cc/) which uses yaml file to set arguments for python scripts. 
These config files can be combined to be easier to read and that's exactly what we have done.
If you check the "configs" directory, you'll see three subdirectories and a bunch of named configuration files.
Each one describes how to run an optimizer for a given problem using some components from the subdirectories - e.g. a specification of a search space or the arguments for the target function.

To get started, we recommend you look at the search spaces and target function configs first. You'll see that these .yaml files describe dictionaries and you can use them to specify whatever arguments or search spaces you want.
The main configs are slightly more complex: we first state which target function, search space and sweeper to use, then we configure the optimizer itself.

The configuration of the optimizers looks quite different, you'll notice, which is simply due to each optimizer's implementation. That means: *to use the hypersweeper, you'll need to know how to use the optimizer you're trying to parallelize!* Random search is an example of a very minimal config - SMAC has a lot more options. Go through two or three examples to see the common options and also the differences. If you want to use an optimizer from the examples, often you can simply copy and modify the config file slighty, but you should still be aware where different keywords would go.

## Running an Example

We pre-configured a few different combinations for you. These configuration files have the naming scheme "<target_function>_<optimizer>.yaml". To run one of them, you'll need to run the following command:

```bash
python <target_function>.py -m --config-name=<target_function>_<optimizer>
```

The "-m" flag will ensure that the hypersweeper is run instead of just executing a single run of the target function. 

If you want to run some variation of our pre-configured examples, e.g. increasing the buget, you can do this in the command line - e.g. for smac on the MLP:

```bash
python mlp.py -m --config-name=mlp_smac hydra.sweeper.n_trials=20
```

You can see the key naming in the configuration files themselves. Obviously you can also make a new file with the settings you prefer - in fact we encourage you to try this out for yourself.

## Inspecting the results

Your results will be localed in a "tmp" directory after executing a run. All your runs will be ordered by date and then time. Once you check the run directory you want to inspect, you'll see the following component:
- a number of numbered directories
- a .csv file containing the runhistory
- a .yaml file with the final config
- (optional) a checkpoint directory
- (optional) optimizer logs

Each of the directories contains the files from a single run - the numbering represents the order they were run in. That means if you write per-run data, this is where you can recover it. The checkpoint directory, on the other hand, is reserved for checkpoints we want to load later using the hypersweeper. 
Usually, the most important files, however, will be the final config and runhistory. The final config can directly be used to execute the best found run again and again. 
The runhistory contains all configurations with their budgets and performances so you can inspect the optimization process. 

## Post-Processing with DeepCave
You can transfer the results to DeepCave fairly easily for further analysis. We prepared a notebook in this directory to demonstrate it and you should be able to use the same function for your future runs as well.