from torchvision.models import *

def get_network(net_name = 'convnext_tiny', num_classes = 5):

    if net_name == 'convnext_tiny':
        return convnext_tiny(num_classes = num_classes, stochastic_depth_prob = 0.1, layer_scale = 1e-6)
    elif net_name == 'convnext_small':
        return convnext_small(num_classes = num_classes, stochastic_depth_prob = 0.4, layer_scale = 1e-6)    
    elif net_name == 'convnext_base':
        return convnext_base(num_classes = num_classes, stochastic_depth_prob = 0.5, layer_scale = 1e-6)   
    elif net_name == 'convnext_large':
        return convnext_large(num_classes = num_classes, stochastic_depth_prob = 0.5, layer_scale = 1e-6)
        