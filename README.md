# Overview:
This mathematically justified framework offers a data-driven approach to
uncertainty quantification for Bayesian inverse problems. When a neural network
comes into play, the information contained within a training dataset is embedded into the
network; the output of which quantifies the uncertainty in the underlying
parameter estimation problem using this information.

# Paper:
The paper associated with this code is titled
"Solving Bayesian Inverse Problems via Variational Autoencoders"
and can be found at: https://arxiv.org/abs/1912.04212

# Installation:
In the `uq-vae` directory, run `conda env create --file environment.yml`. Then
activate `uq_vae_env` to utilize the created environment.

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
* Contains project specific wrappers and routines. Note that the drivers in `/projects/project_name/drivers_/` are agnostic to the project.
  Project dependent codes are all stored in `/projects/project_name/utils_project/`
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
       `/codes/projects/test_discrete_parameter/data_generator/`, use `generate_data.py`
       to generate a training dataset of size 5000 and a testing dataset of size
       200 by setting the generate_train_data and generate_test_data booleans as
       well as the num_data value in the Options class.
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
    3. Predict and plot using the trained neural network.\
       In
        `/codes/projects/test_discrete_parameter/drivers_vae_full/`, run
       `prediction_and_plotting_vae_full.py`. Your plots should be stored in the
       `uq-vae/figures/` directory.\
       Note that you may not see anything too
       interesting in the bivariate plots. This depends on the sample of the
       testing set you're visualizing. To play around with different samples,
       in
       `/test_discrete_parameter/utils_project/prediction_and_plotting_routine_vae_full.py`
       simply change the value of sample_number.

# Contact:
If you have any questions, please feel free to contact me at Hwan.Goh@gmail.com
and I'll reply as soon as I can!
