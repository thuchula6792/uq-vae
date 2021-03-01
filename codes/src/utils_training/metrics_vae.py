import tensorflow as tf
import numpy as np
import pandas as pd

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

###############################################################################
#                    Initialize Metrics and Storage Arrays                    #
###############################################################################
class Metrics:
    def __init__(self):
        #=== Metrics ===#
        self.mean_loss_train = tf.keras.metrics.Mean()
        self.mean_loss_train_vae = tf.keras.metrics.Mean()
        self.mean_loss_train_encoder = tf.keras.metrics.Mean()
        self.mean_loss_train_posterior = tf.keras.metrics.Mean()
        self.mean_loss_train_prior = tf.keras.metrics.Mean()

        self.mean_loss_val = tf.keras.metrics.Mean()
        self.mean_loss_val_vae = tf.keras.metrics.Mean()
        self.mean_loss_val_encoder = tf.keras.metrics.Mean()
        self.mean_loss_val_posterior = tf.keras.metrics.Mean()
        self.mean_loss_val_prior = tf.keras.metrics.Mean()

        self.mean_loss_test = tf.keras.metrics.Mean()
        self.mean_loss_test_vae = tf.keras.metrics.Mean()
        self.mean_loss_test_encoder = tf.keras.metrics.Mean()
        self.mean_loss_test_posterior = tf.keras.metrics.Mean()
        self.mean_loss_test_prior = tf.keras.metrics.Mean()

        self.mean_relative_error_input_vae = tf.keras.metrics.Mean()
        self.mean_relative_error_latent_posterior = tf.keras.metrics.Mean()
        self.mean_relative_error_input_decoder = tf.keras.metrics.Mean()

        self.relative_gradient_norm = 0

        #=== Initialize Metric Storage Arrays ===#
        self.storage_array_loss_train = np.array([])
        self.storage_array_loss_train_vae = np.array([])
        self.storage_array_loss_train_encoder = np.array([])
        self.storage_array_loss_train_posterior = np.array([])
        self.storage_array_loss_train_prior = np.array([])

        self.storage_array_loss_val = np.array([])
        self.storage_array_loss_val_vae = np.array([])
        self.storage_array_loss_val_encoder = np.array([])
        self.storage_array_loss_val_posterior = np.array([])
        self.storage_array_loss_val_prior = np.array([])

        self.storage_array_loss_test = np.array([])
        self.storage_array_loss_test_vae = np.array([])
        self.storage_array_loss_test_encoder = np.array([])
        self.storage_array_loss_test_posterior = np.array([])
        self.storage_array_loss_test_prior = np.array([])

        self.storage_array_relative_error_input_vae = np.array([])
        self.storage_array_relative_error_latent_posterior = np.array([])
        self.storage_array_relative_error_input_decoder = np.array([])

        self.storage_array_relative_gradient_norm = np.array([])

###############################################################################
#                             Update Tensorboard                              #
###############################################################################
    def update_tensorboard(self, summary_writer, epoch):

        with summary_writer.as_default():
            tf.summary.scalar('loss_train',
                    self.mean_loss_train.result(), step=epoch)
            tf.summary.scalar('loss_train_vae',
                    self.mean_loss_train_vae.result(), step=epoch)
            tf.summary.scalar('loss_train_encoder',
                    self.mean_loss_train_encoder.result(), step=epoch)
            tf.summary.scalar('loss_train_posterior',
                    self.mean_loss_train_posterior.result(), step=epoch)
            tf.summary.scalar('loss_train_prior',
                    self.mean_loss_train_prior.result(), step=epoch)

            tf.summary.scalar('loss_val',
                    self.mean_loss_val.result(), step=epoch)
            tf.summary.scalar('loss_val_vae',
                    self.mean_loss_val_vae.result(), step=epoch)
            tf.summary.scalar('loss_val_encoder',
                    self.mean_loss_val_encoder.result(), step=epoch)
            tf.summary.scalar('loss_val_posterior',
                    self.mean_loss_val_posterior.result(), step=epoch)
            tf.summary.scalar('loss_val_prior',
                    self.mean_loss_val_prior.result(), step=epoch)

            tf.summary.scalar('loss_test',
                    self.mean_loss_test.result(), step=epoch)
            tf.summary.scalar('loss_test_vae',
                    self.mean_loss_test_vae.result(), step=epoch)
            tf.summary.scalar('loss_test_encoder',
                    self.mean_loss_test_encoder.result(), step=epoch)
            tf.summary.scalar('loss_test_posterior',
                    self.mean_loss_test_posterior.result(), step=epoch)
            tf.summary.scalar('loss_test_prior',
                    self.mean_loss_test_prior.result(), step=epoch)

            tf.summary.scalar('relative_error_input_vae',
                    self.mean_relative_error_input_vae.result(), step=epoch)
            tf.summary.scalar('relative_error_latent_posterior',
                    self.mean_relative_error_latent_posterior.result(), step=epoch)
            tf.summary.scalar('relative_error_input_decoder',
                    self.mean_relative_error_input_decoder.result(), step=epoch)
            tf.summary.scalar('relative_gradient_norm',
                    self.relative_gradient_norm, step=epoch)

###############################################################################
#                            Update Storage Arrays                            #
###############################################################################
    def update_storage_arrays(self):
        self.storage_array_loss_train =\
                np.append(self.storage_array_loss_train,
                        self.mean_loss_train.result())
        self.storage_array_loss_train_vae =\
                np.append(self.storage_array_loss_train_vae,
                        self.mean_loss_train_vae.result())
        self.storage_array_loss_train_encoder =\
                np.append(self.storage_array_loss_train_encoder,
                        self.mean_loss_train_encoder.result())
        self.storage_array_loss_train_posterior =\
                np.append(self.storage_array_loss_train_posterior,
                        self.mean_loss_train_posterior.result())
        self.storage_array_loss_train_prior =\
                np.append(self.storage_array_loss_train_prior,
                        self.mean_loss_train_prior.result())

        self.storage_array_loss_val =\
                np.append(self.storage_array_loss_val,
                        self.mean_loss_val.result())
        self.storage_array_loss_val_vae =\
                np.append(self.storage_array_loss_val_vae,
                        self.mean_loss_val_vae.result())
        self.storage_array_loss_val_encoder =\
                np.append(self.storage_array_loss_val_encoder,
                        self.mean_loss_val_encoder.result())
        self.storage_array_loss_val_posterior =\
                np.append(self.storage_array_loss_val_posterior,
                        self.mean_loss_val_posterior.result())
        self.storage_array_loss_val_prior =\
                np.append(self.storage_array_loss_val_prior,
                        self.mean_loss_val_prior.result())

        self.storage_array_loss_test =\
                np.append(self.storage_array_loss_test,
                        self.mean_loss_test.result())
        self.storage_array_loss_test_vae =\
                np.append(self.storage_array_loss_test_vae,
                        self.mean_loss_test_vae.result())
        self.storage_array_loss_test_encoder =\
                np.append(self.storage_array_loss_test_encoder,
                        self.mean_loss_test_encoder.result())
        self.storage_array_loss_test_posterior =\
                np.append(self.storage_array_loss_test_posterior,
                        self.mean_loss_test_posterior.result())
        self.storage_array_loss_test_prior =\
                np.append(self.storage_array_loss_test_prior,
                        self.mean_loss_test_prior.result())

        self.storage_array_relative_error_input_vae =\
                np.append(self.storage_array_relative_error_input_vae,
                        self.mean_relative_error_input_vae.result())
        self.storage_array_relative_error_latent_posterior =\
                np.append(self.storage_array_relative_error_latent_posterior,
                        self.mean_relative_error_latent_posterior.result())
        self.storage_array_relative_error_input_decoder =\
                np.append(self.storage_array_relative_error_input_decoder,
                        self.mean_relative_error_input_decoder.result())
        self.storage_array_relative_gradient_norm =\
                np.append(self.storage_array_relative_gradient_norm,
                        self.relative_gradient_norm)

###############################################################################
#                                 Reset Metrics                               #
###############################################################################
    def reset_metrics(self):
        self.mean_loss_train.reset_states()
        self.mean_loss_train_vae.reset_states()
        self.mean_loss_train_encoder.reset_states()
        self.mean_loss_train_posterior.reset_states()
        self.mean_loss_train_prior.reset_states()

        self.mean_loss_val.reset_states()
        self.mean_loss_val_vae.reset_states()
        self.mean_loss_val_encoder.reset_states()
        self.mean_loss_val_posterior.reset_states()
        self.mean_loss_val_prior.reset_states()

        self.mean_loss_test.reset_states()
        self.mean_loss_test_vae.reset_states()
        self.mean_loss_test_encoder.reset_states()
        self.mean_loss_test_posterior.reset_states()
        self.mean_loss_test_prior.reset_states()

        self.mean_relative_error_input_vae.reset_states()
        self.mean_relative_error_latent_posterior.reset_states()
        self.mean_relative_error_input_decoder.reset_states()

###############################################################################
#                                 Save Metrics                                #
###############################################################################
    def save_metrics(self, filepaths):
        metrics_dict = {}
        metrics_dict['loss_train'] = self.storage_array_loss_train
        metrics_dict['loss_train_vae'] = self.storage_array_loss_train_vae
        metrics_dict['loss_train_encoder'] = self.storage_array_loss_train_encoder
        metrics_dict['loss_train_posterior'] = self.storage_array_loss_train_posterior
        metrics_dict['loss_train_prior'] = self.storage_array_loss_train_prior

        metrics_dict['loss_val'] = self.storage_array_loss_val
        metrics_dict['loss_val_vae'] = self.storage_array_loss_val_vae
        metrics_dict['loss_val_encoder'] = self.storage_array_loss_val_encoder
        metrics_dict['loss_val_posterior'] = self.storage_array_loss_val_posterior
        metrics_dict['loss_val_prior'] = self.storage_array_loss_val_prior

        metrics_dict['relative_error_input_vae'] =\
                self.storage_array_relative_error_input_vae
        metrics_dict['relative_error_latent_posterior'] =\
                self.storage_array_relative_error_latent_posterior
        metrics_dict['relative_error_input_decoder'] =\
                self.storage_array_relative_error_input_decoder
        metrics_dict['relative_gradient_norm'] = self.storage_array_relative_gradient_norm

        df_metrics = pd.DataFrame(metrics_dict)
        df_metrics.to_csv(filepaths.trained_NN + "_metrics" + '.csv', index=False)
