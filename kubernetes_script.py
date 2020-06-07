"""Kubernetes script"""
# Training initial models
# from brainreader import train
# from brainreader import params

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

# train.TrainedModel.populate({'dset_id': 1, 'data_params': 3, 'model_params': 4},
#                             'training_params >132 AND training_params <=144',
#                             reserve_jobs=True)  # weighted poisson

# train.TrainedModel.populate({'dset_id': 1, 'data_params': 3}, 'model_params in (29, 30)',
#                             'training_params >72 AND training_params <=84',
#                             reserve_jobs=True)  # konstinets

# train.TrainedModel.populate({'dset_id': 1, 'data_params': 3, 'training_params': 145},
#                             'model_params in (29, 30)', reserve_jobs=True)  # konstinets



# from brainreader.encoding import train
# train.TrainedModel.populate('dset_id in (1, 5)', reserve_jobs=True)
# train.Evaluation.populate(reserve_jobs=True)
# train.Ensemble.populate(reserve_jobs=True)
# train.EnsembleEvaluation.populate(reserve_jobs=True)

# train Gabor
# from brainreader import decoding
# decoding.GaborModel.populate('dset_id in (1, 5)', reserve_jobs=True)

# train AHP
# from brainreader import reconstructions
# reconstructions.AHPValEvaluation.populate(reserve_jobs=True)

# from brainreader import reconstructions
# reconstructions.GradientOneReconstruction.fill_recons('ensemble_dset=5 AND gradient_params > 200', split='val')

# Populate all models for dataset 4
from brainreader import decoding
# decoding.LinearModel.populate({'dset_id': 4}, reserve_jobs=True)
# decoding.LinearValEvaluation.populate({'dset_id': 4}, reserve_jobs=True)
# decoding.LinearReconstructions.populate({'dset_id': 4}, reserve_jobs=True)
# decoding.LinearEvaluation.populate({'dset_id': 4}, reserve_jobs=True)

# decoding.MLPModel.populate({'dset_id': 4}, reserve_jobs=True)
# decoding.MLPValEvaluation.populate({'dset_id': 4}, reserve_jobs=True)
# decoding.MLPReconstructions.populate({'dset_id': 4}, reserve_jobs=True)
# decoding.MLPEvaluation.populate({'dset_id': 4}, reserve_jobs=True)

decoding.GaborModel.populate({'dset_id': 4}, reserve_jobs=True)
decoding.GaborValEvaluation.populate({'dset_id': 4}, reserve_jobs=True)
decoding.GaborReconstructions.populate({'dset_id': 4}, reserve_jobs=True)
decoding.GaborEvaluation.populate({'dset_id': 4}, reserve_jobs=True)

# from brainreader.encoding import train
# train.TrainedModel.populate({'dset_id': 4}, reserve_jobs=True)
# train.Evaluation.populate({'dset_id': 4}, reserve_jobs=True)
# train.Ensemble.populate({'ensemble_dset': 4}, reserve_jobs=True)
# train.EnsembleEvaluation.populate({'ensemble_dset': 4}, reserve_jobs=True)

from brainreader import reconstructions
# Use val_corr in EnsembleEvaluation to add an entry in reconstructions.BestEnsemble
reconstructions.ModelResponses.populate({'ensemble_dset': 4}, reserve_jobs=True) # needs to be done only once per dset
reconstructions.AHPValEvaluation.populate({'ensemble_dset': 4}, reserve_jobs=True)
reconstructions.AHPEvaluation.populate({'ensemble_dset': 4}, reserve_jobs=True)