""" PyTorch dataset. """
from torch.utils import data
import torch

class EncodingDataset(data.Dataset):
    """ A dataset with examples to train an encoding model,

    Arguments:
        images: A N x height x width np.array.
        responses: A N x num_cells np.array

    Returns:
        A (image, responses) tuple:
            image: A 1 x h x w float tensor
            responses: A num_cells float tensor
    """

    def __init__(self, images, responses):
        if len(images) != len(responses):
            raise ValueError('Images and responses should have the same length')
        self.images = images
        self.responses = responses
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        im = torch.as_tensor(self.images[index][None, ...], dtype=torch.float32)
        responses = torch.as_tensor(self.responses[index], dtype=torch.float32)
        
        return im, responses