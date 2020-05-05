""" All trained models """
import datajoint as dj
from torch.utils import data
import torch
from torch import optim
from torch.optim import lr_scheduler
import copy
import time
from torch.nn import functional as F
import numpy as np

from brainreader import data as brdata
from brainreader import params
from brainreader import datasets
from brainreader import utils

schema = dj.schema('br_train')
dj.config["enable_python_native_blobs"] = True  # allow blob support in dj 0.12
dj.config['stores'] = {
    'brdata': {'protocol': 'file', 'location': '/mnt/scratch07/ecobost'}, }
dj.config['cache'] = '/tmp'


def compute_correlation(resps1, resps2):
    """ Compute average correlation between two sets of responses.
    
    Computes correlation per cell (i.e., across images) and returns the average over 
    cells.
    
    Arguments:
        resps1, resps2 (torch.tensor): A (num_images, num_cells) tensor.
    
    Returns:
        corr (float): Average correlation coefficient.
    
    Note:
        Not differentiable.
    """
    corrs = utils.compute_correlation(resps1.detach().cpu().numpy().T,
                                      resps2.detach().cpu().numpy().T)

    # Check that all corrs are valid
    bad_cells = (corrs < -1) | (corrs > 1) | np.isnan(corrs) | np.isinf(corrs)
    if np.any(bad_cells):
        print('Warning: Unstable correlation (setting to -1).', flush=True)
        corrs[bad_cells] = -1

    return corrs.mean()

@schema
class TrainedModel(dj.Computed):
    definition = """ # a single trained model
    
    -> brdata.Responses
    -> params.DataParams
    -> params.ModelParams
    -> params.TrainingParams
    ---
    train_losses:   longblob        # training loss per batch
    train_corrs:    longblob        # training correlation per batch
    val_losses:     longblob        # validation loss per epoch
    val_corrs:      longblob        # validation correlation per epoch
    lr_history:     longblob        # learning rate per epoch
    # diverged:       boolean         # whether loss diverged during training
    best_model:     blob@brdata     # state dictionary for the best model
    best_loss:      float           # validation loss for the best model
    best_corr:      float           # validation correlation for the best model
    best_epoch:     smallint        # epoch were the best model was found
    training_time:  smallint        # how many minutes it took to train this network
    training_ts=CURRENT_TIMESTAMP: timestamp
    """
    # @property
    # def key_source(self):
    #     """ Restrict to only the models that work well. """
    #     all_keys = (brdata.Responses * params.DataParams * params.ModelParams *
    #                 params.TrainingParams)
    #     return all_keys & {'data_params': 1, 'model_params': 2} & 'training_params <= 60'

    @staticmethod
    def _compute_loss(pred_responses, responses, loss_function='mse'):
        """ Computes the appropiate loss function. Differentiable
        
        Arguments:
            pred_responses, responses (tensor): A (num_images, num_cells) tensor with cell
                responses.
            loss_function (string): What loss function to use:
                'mse': Mean squared error.
                'poisson': Poisson loss.
        
        Returns:
            loss (float): Value of the loss function for current predictions.
        """
        if loss_function == 'mse':
            loss = F.mse_loss(pred_responses, responses)
        elif loss_function == 'poisson':
            loss = F.poisson_nll_loss(pred_responses, responses, log_input=False)
        elif loss_function == 'exp': # nll for an exponential curve
            loss = torch.log(pred_responses) + responses / (pred_responses + 1e-8)
            loss = loss.mean()
            djdkdl
        else:
            raise NotImplementedError(f'Loss function {loss_function} not implemented')

        return loss

    def make(self, key):
        utils.log('Training model {model_params} in dataset {dset_id} with data params '
                  '{data_params} and training params {training_params}'.format(**key))
        train_params = (params.TrainingParams & key).fetch1()

        # Set random seed
        torch.manual_seed(train_params['seed'])
        torch.backends.cudnn.deterministic = True

        # Get data
        utils.log('Fetching data')
        dset_id = key['dset_id']
        train_images = (params.DataParams & key).get_images(dset_id, split='train')
        train_responses = (params.DataParams & key).get_responses(dset_id, split='train')
        train_dset = datasets.EncodingDataset(train_images, train_responses)
        train_dloader = data.DataLoader(train_dset, batch_size=train_params['batch_size'],
                                        shuffle=True, num_workers=2)

        val_images = (params.DataParams & key).get_images(dset_id, split='val')
        val_responses = (params.DataParams & key).get_responses(dset_id, split='val')
        val_dset = datasets.EncodingDataset(val_images, val_responses)
        val_dloader = data.DataLoader(val_dset, batch_size=128, num_workers=2)

        # Get model
        utils.log('Instantiating model')
        num_cells = train_responses.shape[-1]
        model = (params.ModelParams & key).get_model(num_cells)
        model.init_parameters()
        model.train()
        model.cuda()



        # TODO: DELETE!
        if train_params['momentum'] < 0:
            print('Using ADAM')
            optimizer = optim.Adam(model.parameters(), lr=float(train_params['learning_rate']),
                              weight_decay=float(train_params['weight_decay']))
        else:
            print('USING SGD')
            optimizer = optim.SGD(model.parameters(), lr=float(train_params['learning_rate']),
                                momentum=float(train_params['momentum']), nesterov=True,
                                weight_decay=float(train_params['weight_decay']))


        # Declare optimizer
        # optimizer = optim.SGD(model.parameters(), lr=float(train_params['learning_rate']),
        #                       momentum=float(train_params['momentum']), nesterov=True,
        #                       weight_decay=float(train_params['weight_decay']))
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                   factor=float(train_params['lr_decay']),
                                                   patience=int(round(
                                                       train_params['decay_epochs'] /
                                                       train_params['val_epochs'])),
                                                   verbose=True)

        # Initialize some logs
        train_losses = []
        train_corrs = []
        val_losses = []
        val_corrs = []
        lr_history = []
        best_model = copy.deepcopy(model).cpu()
        best_epoch = 0
        best_corr = -1
        best_loss = float('inf')
        start_time = time.time()  # in seconds

        # Train
        for epoch in range(1, train_params['num_epochs'] + 1):
            utils.log(f'Epoch {epoch}:')

            # Record learning rate
            lr_history.append(optimizer.param_groups[0]['lr'])

            # Loop over training set
            for images, responses in train_dloader:
                # Zero the gradients
                model.zero_grad()

                # Move variables to GPU
                images, responses = images.cuda(), responses.cuda()

                # Forward
                pred_responses = model(images)

                # Compute loss
                loss = TrainedModel._compute_loss(responses, pred_responses, 
                                                  train_params['loss_function'])

                # Check for divergence
                if torch.isnan(loss) or torch.isinf(loss):
                    raise ValueError('Loss diverged')

                # Backprop
                loss.backward()
                optimizer.step()

                # Compute correlation
                corr = compute_correlation(responses, pred_responses)

                # Record training losses
                train_losses.append(loss.item())
                train_corrs.append(corr)
                utils.log('Training loss (correlation): {:.5f} ({:.3f})'.format(
                    loss.item(), corr))

            # Compute validation metrics and save best model
            if epoch % train_params['val_epochs'] == 0:
                # Compute loss and correlation on validation set
                with torch.no_grad():
                    model.eval()
                    all_resps = [(r.cuda(), model(im.cuda())) for im, r in val_dloader]
                    responses = torch.cat([r[0] for r in all_resps])
                    pred_responses = torch.cat([r[1] for r in all_resps])
                    model.train()

                    val_loss = TrainedModel._compute_loss(responses, pred_responses, 
                                                          train_params['loss_function'])
                    val_corr = compute_correlation(responses, pred_responses)

                # Check for divergence
                if torch.isnan(val_loss) or torch.isinf(val_loss):
                    raise ValueError('Validation loss diverged')

                # Record validation loss
                val_losses.append(val_loss.item())
                val_corrs.append(val_corr)
                utils.log('Validation loss (correlation): {:.5f} ({:.3f})'.format(
                    val_loss.item(), val_corr))

                # Reduce learning rate
                scheduler.step(val_corr)

                # Save best model yet (if needed)
                if val_corr > best_corr:
                    utils.log('Saving best model')
                    best_corr = val_corr
                    best_epoch = epoch
                    best_loss = val_loss.item()
                    best_model = copy.deepcopy(model).cpu()

                # Stop training if validation has not improved in x number of epochs
                if epoch - best_epoch >= train_params['stopping_epochs']:
                    utils.log('Stopping training. Validation has not improved in '
                              '{stopping_epochs} epochs.'.format(**train_params))
                    break

        # Clean up (if needed)
        training_time = round((time.time() - start_time) / 60) # in minutes
        utils.log(f'Finished training in {training_time} minutes')

        # Insert results
        utils.log('Inserting results')
        results = key.copy()
        results['train_losses'] = np.array(train_losses, dtype=np.float32)
        results['train_corrs'] = np.array(train_corrs, dtype=np.float32)
        results['val_losses'] = np.array(val_losses, dtype=np.float32)
        results['val_corrs'] = np.array(val_corrs, dtype=np.float32)
        results['lr_history'] = np.array(lr_history, dtype=np.float32)
        #results['diverged'] = False
        results['best_model'] = {k: v.cpu().numpy() for k, v in
                                 best_model.state_dict().items()}
        results['best_loss'] = best_loss
        results['best_corr'] = best_corr
        results['best_epoch'] = best_epoch
        results['training_time'] = training_time
        self.insert1(results)


    def get_model(self):
        """ Load a trained model."""
        # Find num_cells for this dataset (with its data_params)
        dset_id = self.fetch1('dset_id')
        cell_mask = (params.DataParams & self).get_cell_mask(dset_id)
        num_cells = np.count_nonzero(cell_mask)

        # Get model
        model = (params.ModelParams & self).get_model(num_cells)

        # Load saved weights
        state_dict = {k: torch.as_tensor(v) for k, v in self.fetch1('best_model').items()}
        model.load_state_dict(state_dict)

        return model


@schema
class Evaluation(dj.Computed):
    definition = """ # evaluate models in test set
    -> TrainedModel
    ---
    test_corr:  float
    """
    def make(self, key):
        # Get data
        dset_id = key['dset_id']
        test_images = (params.DataParams & key).get_images(dset_id, split='test')
        test_responses = (params.DataParams & key).get_responses(dset_id, split='test')
        test_dset = datasets.EncodingDataset(test_images, test_responses)
        test_dloader = data.DataLoader(test_dset, batch_size=128, num_workers=4)

        # Get model
        model = (TrainedModel & key).get_model()
        model.eval()
        model.cuda()

        # Compute correlation
        with torch.no_grad():
            all_resps = [(r.cuda(), model(im.cuda())) for im, r in test_dloader]
            responses = torch.cat([r[0] for r in all_resps])
            pred_responses = torch.cat([r[1] for r in all_resps])

            corr = compute_correlation(responses, pred_responses)

        # Insert
        self.insert1({**key, 'test_corr': corr})


@schema
class Ensemble(dj.Computed):
    definition = """ # group of models that form an ensemble
    -> TrainedModel.proj(ensemble_dset='dset_id', ensemble_data='data_params', ensemble_model='model_params', ensemble_training='training_params') # one of the models in the ensemble (always the same for the same ensemble params)
    -> params.EnsembleParams
    ---
    num_models:         tinyint # number of models in this ensemble
    ensemble_ts=CURRENT_TIMESTAMP:  timestamp
    """

    class OneModel(dj.Part):
        definition = """ # a single model that belongs to this ensemble
        -> master
        -> TrainedModel
        """

    @property
    def key_source(self):
        """ TrainedModel key (* EnsembleParams) of one of the models of each ensemble.
        
        For ensemble_params=1 (i.e. ensemble all the models with the same config but 
        different seed), we return the model of each config with seed=1234.
        """
        all_keys = TrainedModel * params.EnsembleParams
        restr_keys = all_keys & {'ensemble_params': 1} & (params.TrainingParams &
                                                          {'seed': 1234})
        keys = restr_keys.proj('ensemble_params', ensemble_dset='dset_id',
                               ensemble_data='data_params', ensemble_model='model_params',
                               ensemble_training='training_params')
        return keys

    def make(self, key):
        if key['ensemble_params'] == 1:
            # Find all training configs that differ only in seed
            training_params = (params.TrainingParams &
                               {'training_params': key['ensemble_training']}).fetch1()
            del training_params['training_params'] # ignore the specific training params
            del training_params['seed'] # ignore seed
            training_keys = (params.TrainingParams & training_params).proj()

            # Find all trained models with the desired config
            key_wo_training = {'dset_id': key['ensemble_dset'],
                               'data_params': key['ensemble_data'],
                               'model_params': key['ensemble_model']}
            models = TrainedModel & key_wo_training & training_keys

            # Check that five models with this config have been trained
            num_models = len(models)
            if num_models != 5:
                raise ValueError('Expected to have five seeds per model config.')

            # Insert
            self.insert1({**key, 'num_models': num_models})
            for one_key in models.proj():
                self.OneModel.insert1({**key, **one_key})
        else:
            msg = 'Ensemble method {ensemble_params} not implemented'.format(**key)
            raise NotImplementedError(msg)

    def get_model(self):
        """ Loads all models in this ensemble and creates an Ensemble model from it."""
        from brainreader import models

        # Get models
        models_ = [(TrainedModel & key).get_model() for key in (Ensemble.OneModel & self)]

        # Create Ensemble model
        ensemble = models.Ensemble(models_)

        return ensemble


@schema
class EnsembleEvaluation(dj.Computed):
    definition = """ # evaluate an ensemble of models
    -> Ensemble
    ---
    val_corr:       float
    test_corr:      float
    """
    def make(self, key):
        # Get validation data
        dset_id = (dj.U('dset_id') & (Ensemble.OneModel & key)).fetch1('dset_id')
        dataparams = params.DataParams & (Ensemble.OneModel & key)
        val_images = dataparams.get_images(dset_id, split='val')
        val_responses = dataparams.get_responses(dset_id, split='val')
        val_dset = datasets.EncodingDataset(val_images, val_responses)
        val_dloader = data.DataLoader(val_dset, batch_size=128, num_workers=4)

        # Get test data
        test_images = dataparams.get_images(dset_id, split='test')
        test_responses = dataparams.get_responses(dset_id, split='test')
        test_dset = datasets.EncodingDataset(test_images, test_responses)
        test_dloader = data.DataLoader(test_dset, batch_size=128, num_workers=4)

        # Get model
        model = (Ensemble & key).get_model()
        model.eval()
        model.cuda()

        # Compute correlation
        with torch.no_grad():
            # Compute validation correlation
            all_resps = [(r.cuda(), model(im.cuda())) for im, r in val_dloader]
            responses = torch.cat([r[0] for r in all_resps])
            pred_responses = torch.cat([r[1] for r in all_resps])
            val_corr = compute_correlation(responses, pred_responses)

            # Compute test correlation
            all_resps = [(r.cuda(), model(im.cuda())) for im, r in test_dloader]
            responses = torch.cat([r[0] for r in all_resps])
            pred_responses = torch.cat([r[1] for r in all_resps])
            test_corr = compute_correlation(responses, pred_responses)

        # Insert
        self.insert1({**key, 'val_corr': val_corr, 'test_corr': test_corr})


# this works but we don't need it
# @schema
# class AverageEvaluationInEnsemble(dj.Computed):
#     definition = """ # takes correlation from each model in an ensemble and averages them
#     -> Ensemble
#     ---
#     val_corr:       float       # average val correlation across all models in this ensemble
#     test_corr:      float       # average test correlation across all models in this ensemble
#     val_corrs:      longblob    # all validation correlations in this ensemble
#     test_corrs:     longblob    # all test correlations in this ensemble
#     std_val_corrs:  float       # standard deviation across validation correlations
#     std_test_corrs: float       # standard deviation across test correlations
#     """

#     def make(self, key):
#         """
#         Ensemble evaluation averages the model responses and computes correlations
#         afterwards, this averges the single model correlations computed in Evaluation. It
#         does not do any processing.
#         """
#         # Get corrs
#         models = (Ensemble.OneModel & key)
#         val_corrs, test_corrs = (TrainedModel * Evaluation & models).fetch('best_corr',
#                                                                            'test_corr')

#         # Insert
#         self.insert1({**key, 'val_corr': val_corrs.mean(), 'val_corrs': val_corrs,
#                       'std_val_corrs': val_corrs.std(), 'test_corr': test_corrs.mean(),
#                       'test_corrs': test_corrs, 'std_test_corrs': test_corrs.std()})


#TODO:
# @schema
# class EnsembleCellEvaluation(dj.Computed):
#     definition = """ # test correlations per cell
#     -> Ensemble
#     ---
#     val_corrs:      longblob        # validation correlation for each cell in the model
#     test_corrs:     longblob        # validation correlation for each cell in the model
#     """
#     def make(self, key):
#         # TODO: Copy from EnsembleEvaluation and maybe modify compute_correlation to return results per cell
#         # TODO: Maybe add fraction oracle scores here too?
