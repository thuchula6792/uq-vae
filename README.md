# Paper:
* The paper associated with this code is titled
  "Solving Bayesian Inverse Problems via Variational Autoencoders"
  and can be found at: https://arxiv.org/abs/1912.04212

# Code Structure:
* Below is a description of the source codes as well as the key codes in my
  input/output system.
* For an illustrative example, see the test cases in `codes/projects/test_`. To run
  this, use any of the training drivers in `codes/projects/test_/training_.py`. You may
  have to first generate the training and testing data using
  `codes/projects/test_/data_generator/generate_data.py`.
* However, there is no need to use my input/output system; you can use
  your own input/output codes so long as the appropriate calls to the neural
  networks in `src/neural_networks` and optimization routines in `src/optimize` are
  implemented.

## src:
* `data_handler.py`:                    Class that loads and processes data
* `form_train_val_test_tf_batches.py`:  Form training, validation and test batches
                                        from loaded data using Tensorflow's Dataset
                                        API
* `nn_.py`:                             The neural network
* `functionals.py`:                     Functionals that form the overall loss
                                        functional
* `optimize_.py`:                       The optimization routine for the neural network
* `metrics_.py`:                        Class storing and updating the optimization information
* `get_hyperparameter_combinations.py`: Forms combinations of the hyperparameters
                                        for scheduled routines
* `schedule_and_run.py`:                Uses hyperparameter combinations to run a distributed
                                        schedule of training routines using mpi4py
* `file_paths_.py`:                     Class containing file paths for neural
                                        network and training related objects

## projects:
* Contains project specific wrappers and routines. Note that the drivers are agnostic to the project. Project dependent codes are all stored in utils_project
* drivers:
    * `training_.py`:                  Drives the training routine. Consists of the
                                       Hyperparameter class and calls the FilePaths class and the training_routine
                                       method
    * `hyperparameter_optimization_.py`: Drives hyperparameter optimization for
                                       training neural networks. Utilizes scikit-optimize`s
                                       Bayesian optimization routines.
    * `prediction_and_plotting_.py`:   Drives the prediction and plotting routine given a trained neural
                                       network
    * `scheduler_training_.py`:        Drives the formation of hyperparameter combinations
                                       and schedule of training routines using mpi4py
* utils_project:
	* `file_paths_project.py`:  Class containing file paths for project specific objects
    * `construct_data_dict.py`: Constructs dictionary storing processed data and
                                data related objects to be passed to be the
                                training routine
	* `training_routine_.py`:   Loads the data, constructs the neural
                                network and runs the optimization routine
	* `hyperparameter_optimization_training_routine_.py`: Optimization
                                routine for Bayesian hyperparameter
                                optimization
	* `prediction_and_plotting_routine.py`:  Prediction routine; using trained network,
                                form and save predictions given trained
                                neural network

* config_files:
    * `hyperparameters_.yaml`: YAML file containing hyperparameters for training
                               the neural network
    * `options_.yaml`:         YAML file containing the options for the project
                               as well as the neural network
