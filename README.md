# Paper:
The paper associated with this code is titled
"Solving Bayesian Inverse Problems via Variational Autoencoders"
and can be found at: https://arxiv.org/abs/1912.04212

# Overview:
We developed a variational inference based theoretical framework for uncertainty
quantification in Bayesian inverse problems. Implementability of this theory is
achieved through consideration of amortized inference over a dataset which, in
turn, naturally lends itself to reparameterization by a neural network. The
resulting variational autoencoder is able to perform rapid Bayesian inference on
the parameters of interest.

# Code Structure:
* Below is a description of the source codes as well as the key codes in my
  input/output system.
* The design philosophy underlying my input/output system is based on the clear
  separation of the project and the neural network. Indeed:
    * The scripts in `/projects/utils_project/` oversee the
      importation and preparation of the project specific objects for
      optimization.
    * The scripts in `/src/` oversee the optimization and exportation
      of a neural network. These scripts are agnostic to the project; they
      maintain an arbitrary view of the project specific objects.
* However, there is no need to use my input/output system. You can use
  your own input/output codes so long as the appropriate calls to the neural
  networks in `/src/neural_networks/` and optimization routines in `/src/optimize/` are
  implemented.

## src:
* `/utils_io/file_paths_.py`:           Class containing file paths for neural
                                        network and training related objects
* `/utils_data/data_handler.py`:        Class that loads and processes data
* `/utils_training/form_train_val_test_tf_batches.py`:  Form training, validation and test batches
                                        from loaded data using Tensorflow's Dataset
                                        API
* `/neural_networks/nn_.py`:            The neural network
* `/utils_training/functionals.py`:     Functionals that form the overall loss
                                        functional
* `/optimize/optimize_.py`:             The optimization routine for the neural network
* `/utils_training/metrics_.py`:        Class storing and updating the optimization information
* `/utils_hyperparameter_optimization/get_hyperparameter_combinations.py`: Forms combinations of the hyperparameters
                                        for scheduled routines
* `/utils_scheduler/schedule_and_run.py`: Uses hyperparameter combinations to run a distributed
                                        schedule of training routines using mpi4py

## projects:
* Contains project specific wrappers and routines. Note that the drivers in `/project/drivers_/` are agnostic to the project.
  Project dependent codes are all stored in `/project/utils_project/`
* `/drivers_/`:
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
* `/utils_project/`:
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

* `/config_files/`:
    * `hyperparameters_.yaml`: YAML file containing hyperparameters for training
                               the neural network
    * `options_.yaml`:         YAML file containing the options for the project
                               as well as the neural network

# Illustrative Example:
* To run an illustrative example, please follow these steps:
    1. Generate the data.\
       In
       `/codes/projects/test_discrete_parameter/data_generator/`, run `generate_data.py`.
       Your data should now be generated into the `datasets` directory at the
       same level as the `uq-vae` directory.
    2. Train the neural network.\
       In
       `/codes/projects/test_discrete_parameter/drivers_vae_full/`, run
       `training_vae_full_linear_model_augmented_autodiff.py`. Your trained
       neural network should be stored in the `uq-vae/trained_nns/` directory
       and the tracked Tensorboard metrics in the `uq-vae/tensorboard/`
       directory. To view the Tensorboard metrics, use 'tensorboard
       --logdir=tensorboard' while in the uq-vae directory and click on the
       generated link.
    3. Predict and plot using trained neural network.\
       In
        `/codes/projects/test_discrete_parameter/drivers_vae_full/`, run
       `prediction_and_plotting_vae_full.py`. Your plots should be stored in the
       `uq-vae/figures/` directory.

# Contact:
If you have any questions, please feel free to contact me at Hwan.Goh@gmail.com
and I'll reply as soon as I can!
