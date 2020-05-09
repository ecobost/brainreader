"""Kubernetes script"""
# Training initial models
from brainreader import train
from brainreader import params


# train.TrainedModel.populate({'dset_id': 1, 'model_params': 1, 'data_params': 1}, 'training_params <=6',
#                             reserve_jobs=True) # MSE + none

# train.TrainedModel.populate({'dset_id': 1, 'model_params': 3, 'data_params': 3},
#                             'training_params <=6', reserve_jobs=True)  # MSE + elu

# train.TrainedModel.populate({'dset_id': 1, 'model_params': 4, 'data_params': 3},
#                             'training_params >6 AND training_params <=12',
#                             reserve_jobs=True)  # MSE + expscaled


# train.TrainedModel.populate({'dset_id': 1, 'model_params': 2, 'data_params': 3},
#                             'training_params >12 AND training_params <=18',
#                             reserve_jobs=True)  # poisson + exp

# train.TrainedModel.populate({'dset_id': 1, 'model_params': 3, 'data_params': 3},
#                             'training_params >18 AND training_params <=24',
#                             reserve_jobs=True)  # poisson + elu

# train.TrainedModel.populate({'dset_id': 1, 'model_params': 4, 'data_params': 3},
#                             'training_params >18 AND training_params <=24',
#                             reserve_jobs=True)  # poisson + expscaled


# train.TrainedModel.populate({'dset_id': 1, 'model_params': 2, 'data_params': 3},
#                             'training_params >24 AND training_params <=30',
#                             reserve_jobs=True)  # exp + exp

# train.TrainedModel.populate({'dset_id': 1, 'model_params': 3, 'data_params': 3},
#                             'training_params >24 AND training_params <=30',
#                             reserve_jobs=True)  # exp + elu

# train.TrainedModel.populate({'dset_id': 1, 'model_params': 4, 'data_params': 3},
#                             'training_params >30 AND training_params <=36',
#                             reserve_jobs=True)  # exp + expscaled



# train.TrainedModel.populate({'dset_id': 1, 'model_params': 1, 'data_params': 1},
#                             'training_params >36 AND training_params <=42',
#                             reserve_jobs=True)  # mse + none

# train.TrainedModel.populate({'dset_id': 1, 'model_params': 2, 'data_params': 3},
#                             'training_params >36 AND training_params <=42',
#                             reserve_jobs=True)  # mse + exp

# train.TrainedModel.populate({'dset_id': 1, 'model_params': 3, 'data_params': 3},
#                             'training_params >42 AND training_params <=48',
#                             reserve_jobs=True)  # mse + elu

# train.TrainedModel.populate({'dset_id': 1, 'model_params': 4, 'data_params': 3},
#                             'training_params >36 AND training_params <=42',
#                             reserve_jobs=True)  # mse + expscaled


# train.TrainedModel.populate({'dset_id': 1, 'model_params': 2, 'data_params': 3},
#                             'training_params >48 AND training_params <=54',
#                             reserve_jobs=True)  # poisson + exp

# train.TrainedModel.populate({'dset_id': 1, 'model_params': 3, 'data_params': 3},
#                             'training_params >54 AND training_params <=60',
#                             reserve_jobs=True)  # poisson + elu

# train.TrainedModel.populate({'dset_id': 1, 'model_params': 4, 'data_params': 3},
#                             'training_params >54 AND training_params <=60',
#                             reserve_jobs=True)  # poisson + expscaled


# train.TrainedModel.populate({'dset_id': 1, 'model_params': 2, 'data_params': 3},
#                             'training_params >60 AND training_params <=66',
#                             reserve_jobs=True)  # exp + exp

# train.TrainedModel.populate({'dset_id': 1, 'model_params': 3, 'data_params': 3},
#                             'training_params >60 AND training_params <=66',
#                             reserve_jobs=True)  # exp + elu

# train.TrainedModel.populate({'dset_id': 1, 'model_params': 4, 'data_params': 3},
#                             'training_params >66 AND training_params <=72',
#                             reserve_jobs=True)  # exp + expscaled




# train.TrainedModel.populate({'dset_id': 1, 'model_params': 3, 'data_params': 3},
#                             'training_params >72 AND training_params <=96',
#                             reserve_jobs=True)  # poissson + elu

# train.TrainedModel.populate({'dset_id': 1, 'model_params': 4, 'data_params': 3},
#                             'training_params >72 AND training_params <=96',
#                             reserve_jobs=True)  # poissson + expscaled





# train.TrainedModel.populate({'dset_id': 1, 'model_params': 1, 'data_params': 2},
#                             'training_params <=6', reserve_jobs=True)  # zscore-resps

# train.TrainedModel.populate({'dset_id': 1, 'model_params': 1, 'data_params': 4},
#                             'training_params <=6', reserve_jobs=True)  # df/f

# train.TrainedModel.populate({'dset_id': 1, 'model_params': 1, 'data_params': 5},
#                             'training_params <=6', reserve_jobs=True)  # df/std(df)

# train.TrainedModel.populate({'dset_id': 1, 'model_params': 4, 'data_params': 6},
#                             'training_params >72 AND training_params <=84', reserve_jobs=True)  # stddev-resps



# train.TrainedModel.populate({'dset_id': 1, 'model_params': 4, 'data_params': 3},
#                             'training_params >96 AND training_params <=132',
#                             reserve_jobs=True)  # different batch size


# train.TrainedModel.populate({'dset_id': 1, 'data_params': 3}, 'model_params in (5, 6, 7)',
#                             'training_params >72 AND training_params <=84',
#                             reserve_jobs=True)  # different mlps

train.TrainedModel.populate({'dset_id': 1, 'data_params': 3, 'model_params': 4},
                            'training_params >132 AND training_params <=144',
                            reserve_jobs=True)  # different mlps



# train.TrainedModel.populate({'dset_id': 1, 'model_params': 3 ,'data_params': 1},
#                             'training_params <=16', reserve_jobs=True) # SGD and Adam
# train.TrainedModel.populate({'dset_id': 1, 'data_params': 3, 'model_params': 16},
#                             'training_params > 16 AND training_params < 32', reserve_jobs=True) # poisson
# train.TrainedModel.populate({'dset_id': 1, 'data_params': 1}, 'model_params > 16',
#                             'training_params <= 6', reserve_jobs=True) # smaller MLP
# train.TrainedModel.populate({'dset_id': 1, 'data_params': 1, 'model_params': 19},
#                             'training_params > 16 AND training_params < 32', reserve_jobs=True)  # smaller MLP + poisson (on SGD)

# train.TrainedModel.populate({'dset_id': 1, 'model_params': 20}, 'data_params in (1, 3)',
#                             'training_params <= 6', reserve_jobs=True)  # MSE with elu activation (and either data_params1 or 3)
# train.TrainedModel.populate({'dset_id': 1, 'data_params': 1, 'model_params': 21},
#                              'training_params > 6 AND training_params <=16', reserve_jobs=True) # no batchnorm anywhere (use ADAM, because SGD diverges, probalby needs slower)

# from brainreader import params
# train.TrainedModel.populate({'dset_id': 1, 'data_params': 3, 'model_params': 22},
#                             params.TrainingParams & {'loss_function': 'exp'},
#                             reserve_jobs=True)

# redo poisson
# train.TrainedModel.populate({'dset_id': 1, 'data_params': 3, 'model_params': 22},
#                             params.TrainingParams & {'loss_function': 'poisson', 'momentum': 0.9})

# train.Evaluation.populate(reserve_jobs=True)
# train.Ensemble.populate(reserve_jobs=True)
# train.EnsembleEvaluation.populate(reserve_jobs=True)

# train MLP
# from brainreader import decoding
# decoding.MLPModel.populate('dset_id in (1, 5)', reserve_jobs=True)