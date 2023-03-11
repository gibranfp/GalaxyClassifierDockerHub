import pandas as pd
from torch.utils.data import Dataset
from PIL import Image

from torch_galaxy.transforms import galaxy_transform_train, galaxy_transform_eval
from torch_galaxy.random_center import get_random_center_size

data_path = '../../data/Nair_MaNGA.csv'
images_path = '../images'

'''
# ttype map for 5 clases
ttype_map = {
    0: (-5,-3), 
    1: (-2,0), 
    2: (1,2,3), 
    3: (4,5), 
    4: (6,7,8,9,10)
}
'''

ttype_map = {
    0: (-5,-3), 
    1: (-2,0), 
    2: (1,2), 
    3: (3,4), 
    4: (5,), 
    5: (6,7,8,9,10)
}

def get_label(TType, ttype_map):
    [galaxy_label] = [label for label, interval in ttype_map.items() if TType in interval]
    return galaxy_label

def label_data(data, ttype_map = ttype_map):
    data['label'] = data.apply(lambda galaxy: get_label(galaxy.TType, ttype_map = ttype_map), axis = 1)
    return data

class galaxy_dataset_train(Dataset):

    def __init__(self, dist_info, data_path = data_path, images_path = images_path, transform = galaxy_transform_train):
        
        unlabeled_data = pd.read_csv(data_path)
        self.data = label_data(unlabeled_data)

        self.images_path = images_path

        self.transform = transform

        self.dist_info = dist_info.copy()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        
        img_name = self.data.iloc[index, -2]
        img_path = self.images_path + '/' + img_name
        img      = Image.open(img_path)

        img = self.transform(img, dist_info = self.dist_info)

        label = self.data.iloc[index, -1]

        return img, label

####################################################################

class galaxy_dataset_eval(Dataset):

    def __init__(self, dist_info, data_path = data_path, images_path = images_path, transform = galaxy_transform_eval):
        
        # Read data, label data and add center sizes
        unlabeled_data = pd.read_csv(data_path)
        labeled_data = label_data(unlabeled_data)
        self.data = self.add_center_size(labeled_data, dist_info)

        self.images_path = images_path

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        
        img_name = self.data.iloc[index, -3]
        img_path = self.images_path + '/' + img_name
        img      = Image.open(img_path)

        # Get center size
        center_size = self.data.iloc[index, -1]

        img = self.transform(img, center_size = center_size)

        label = self.data.iloc[index, -2]

        return img_name, img, label

    def add_center_size(self, labeled_data, dist_info):
        # If num_samples == 1 then don't perform center crop
        if dist_info['num_samples'] == 1:
            center_sizes = [None]
        else:
            center_sizes = get_random_center_size(dist_info, case = 'eval')            
        
        center_sizes = pd.DataFrame({'center_size': center_sizes})
        
        return labeled_data.merge(center_sizes, how = 'cross')