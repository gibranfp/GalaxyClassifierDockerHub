from torchvision.transforms import Compose, Resize, ToTensor, RandomHorizontalFlip, RandomVerticalFlip, RandomRotation, CenterCrop
from torch_galaxy.random_center import get_random_center_size

################################################
final_size = 50
#final_size = 300

################################################

rand_flips_and_rotation = Compose([
    RandomHorizontalFlip(), 
    RandomVerticalFlip(), 
    RandomRotation(degrees = (0, 360))])

resize_ = Compose([
    ToTensor(), 
    Resize(final_size)])


def galaxy_transform_train(img, dist_info):
    # First part of the processing
    img = rand_flips_and_rotation(img)
    # Random central crop
    # Train
    center_size = get_random_center_size(dist_info = dist_info, 
                                         case      = 'train')
    #print(center_size)

    # Get custom center crop
    center_crop = CenterCrop(size = center_size)
    
    img = center_crop(img)
    # Second part of the processing
    img = resize_(img)

    return img

def galaxy_transform_eval(img, center_size):
    # Central crop

    if center_size:
        # print(center_size)
        center_crop = CenterCrop(size = center_size)
        img = center_crop(img)
    
    # Second part of the processing
    img = resize_(img)

    return img