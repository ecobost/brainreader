"""
Taken from here:
    https://github.com/aaron-xichen/pytorch-playground/tree/master/mnist
"""
import torch.nn as nn
from collections import OrderedDict
import torch.utils.model_zoo as model_zoo

model_urls = {
    'mnist': 'http://ml.cs.tsinghua.edu.cn/~chenxi/pytorch-models/mnist-b07bb66b.pth'}


class MLP(nn.Module):
    def __init__(self, input_dims, n_hiddens, n_class):
        super(MLP, self).__init__()
        assert isinstance(input_dims, int), 'Please provide int for input_dims'
        self.input_dims = input_dims
        current_dims = input_dims
        layers = OrderedDict()

        if isinstance(n_hiddens, int):
            n_hiddens = [n_hiddens]
        else:
            n_hiddens = list(n_hiddens)
        for i, n_hidden in enumerate(n_hiddens):
            layers['fc{}'.format(i + 1)] = nn.Linear(current_dims, n_hidden)
            layers['relu{}'.format(i + 1)] = nn.ReLU()
            layers['drop{}'.format(i + 1)] = nn.Dropout(0.2)
            current_dims = n_hidden
        layers['out'] = nn.Linear(current_dims, n_class)

        self.model = nn.Sequential(layers)

    def forward(self, input):
        input = input.view(input.size(0), -1)
        assert input.size(1) == self.input_dims
        return self.model.forward(input)


def mnist(input_dims=784, n_hiddens=[256, 256], n_class=10, pretrained=None):
    model = MLP(input_dims, n_hiddens, n_class)
    if pretrained is not None:
        m = model_zoo.load_url(model_urls['mnist'])
        state_dict = m.state_dict() if isinstance(m, nn.Module) else m
        assert isinstance(state_dict, (dict, OrderedDict)), type(state_dict)
        model.load_state_dict(state_dict)
    return model

# Erick: I added this function
import torch
def classify(images, device='cpu'):
    """ Get a model and classify MNIST images.
    
    Arguments:
        images (np.array): A (num_images, 28 x 28) array with images.
        device (torch.device): Where to run the classification.
    
    Returns:
        A np.array with predicted digits for each image.
    """
    # Get model
    model = mnist(pretrained=True)
    model.eval()
    model.to(device)

    # Prepare images
    flat_images = images.reshape(images.shape[0], -1)
    norm_images = (flat_images - 0.1307) / 0.3801
    images = torch.as_tensor(norm_images, dtype=torch.float32, device=device)

    # Run classification
    with torch.no_grad():
        probs = model(images).cpu().numpy()
    pred_labels = probs.argmax(-1)

    return pred_labels
