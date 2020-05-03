"""Kubernetes script"""
#TODO: Make sure that when populating poisson loss I use std normalization and exp activatio

# Training initial models
from brainreader import train
# train.TrainedModel.populate({'dset_id': 1, 'model_params': 3 ,'data_params': 1},
#                             'training_params <=16', reserve_jobs=True) # SGD and Adam
# train.TrainedModel.populate({'dset_id': 1, 'data_params': 3, 'model_params': 16},
#                             'training_params > 16', reserve_jobs=True) # poisson
# train.TrainedModel.populate({'dset_id': 1, 'data_params': 1}, 'model_params > 16',
#                             'training_params <= 6', reserve_jobs=True) # smaller MLP
# train.TrainedModel.populate({'dset_id': 1, 'data_params': 1, 'model_params': 19},
#                             'training_params > 16 AND training_params < 32', reserve_jobs=True)  # smaller MLP + poisson (on SGD)


train.TrainedModel.populate({'dset_id': 1, 'model_params': 20}, 'data_params in (1, 3)',
                            'training_params <= 6', reserve_jobs=True)  # MSE with elu activation (and either data_params1 or 3)
train.TrainedModel.populate({'dset_id': 1, 'data_params': 1, 'model_params': 21},
                             'training_params > 6 AND training_params <=16', reserve_jobs=True) # no batchnorm anywhere

# train.Evaluation.populate(reserve_jobs=True)
# train.Ensemble.populate(reserve_jobs=True)
# train.EnsembleEvaluation.populate(reserve_jobs=True)

# # train MLP
# from brainreader import decoding
# decoding.MLPModel.populate({'dset_id': 1}, reserve_jobs=True)