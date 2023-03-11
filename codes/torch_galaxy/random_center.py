import numpy as np
from scipy.stats import beta

dist_info_base = {'num_samples': 20,
                  'upper_limit': 800,
                  'lower_limit': 300}

##################################################
def random_beta(num_samples, alpha_param, beta_param, upper_limit, lower_limit):

    if num_samples > 1:
        rand_state = 182
    else:
        rand_state = None
    
    # Get random sample
    rand_num = beta.rvs(a            = alpha_param, 
                        b            = beta_param, 
                        loc          = lower_limit, 
                        scale        = (upper_limit - lower_limit), 
                        size         = num_samples,
                        random_state = rand_state)

    # Transform to integers
    rand_num = rand_num.astype(int)

    return rand_num

#################################################
def random_uniform(num_samples, upper_limit, lower_limit):
    
    if num_samples == 1:
        rand_num = np.random.randint(lower_limit, upper_limit + 1, size = 1, dtype = int)
    else:
        rand_num = np.linspace(lower_limit, upper_limit, num_samples, dtype = int)
    
    return rand_num


#################################################
def get_random_center_size(dist_info, case):

    # If case = train return only one
    # random number, else return the amount of
    # num_samples scpecified in dist_info
    if case == 'train':
        num_samples = 1
    elif case == 'eval':
        num_samples = dist_info['num_samples']
    
    if dist_info['dist'] == 'uniform':
        center_sizes = random_uniform(num_samples = num_samples, 
                                      upper_limit = dist_info['upper_limit'],
                                      lower_limit = dist_info['lower_limit'])

    elif dist_info['dist'] == 'beta':
        center_sizes = random_beta(num_samples = num_samples, 
                                   alpha_param = dist_info['alpha_param'], 
                                   beta_param  = dist_info['beta_param'], 
                                   upper_limit = dist_info['upper_limit'], 
                                   lower_limit = dist_info['lower_limit'])

    if num_samples == 1:
        return center_sizes[0]
    else:
        return center_sizes