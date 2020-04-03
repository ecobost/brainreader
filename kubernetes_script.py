"""Kubernetes script"""
#TODO: Make sure that when populating poisson loss I use std normalization and exp activatio

# Training initial models
from brainreader import train
train.TrainedModel.populate(reserve_jobs=True)