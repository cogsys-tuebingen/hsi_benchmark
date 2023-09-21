import numpy as np
import torch
from typing import Optional

from dataloader.dataset_factory import DataObject, get_data

from skimage.morphology import reconstruction
from skimage.morphology import erosion
from skimage.morphology import disk
from skimage import util

def opening_by_reconstruction(image, se):
    """
        Performs an Opening by Reconstruction.

        Parameters:
            image: 2D matrix.
            se: structuring element
        Returns:
            2D matrix of the reconstructed image.
    """
    eroded = erosion(image, se)
    reconstructed = reconstruction(eroded, image)
    return reconstructed


def closing_by_reconstruction(image, se):
    """
        Performs a Closing by Reconstruction.

        Parameters:
            image: 2D matrix.
            se: structuring element
        Returns:
            2D matrix of the reconstructed image.
    """
    obr = opening_by_reconstruction(image, se)

    obr_inverted = util.invert(obr)
    obr_inverted_eroded = erosion(obr_inverted, se)
    obr_inverted_eroded_rec = reconstruction(
        obr_inverted_eroded, obr_inverted)
    obr_inverted_eroded_rec_inverted = util.invert(obr_inverted_eroded_rec)
    return obr_inverted_eroded_rec_inverted


def build_morphological_profiles(image, se_size=4, se_size_increment=2, num_openings_closings=4):
    """
        Build the morphological profiles for a given image.

        Parameters:
            base_image: 2d matrix, it is the spectral information part of the MP.
            se_size: int, initial size of the structuring element (or kernel). Structuring Element used: disk
            se_size_increment: int, structuring element increment step
            num_openings_closings: int, number of openings and closings by reconstruction to perform.
        Returns: 
            emp: 3d matrix with both spectral (from the base_image) and spatial information         
    """
    x, y = image.shape

    cbr = np.zeros(shape=(x, y, num_openings_closings))
    obr = np.zeros(shape=(x, y, num_openings_closings))

    it = 0
    tam = se_size
    while it < num_openings_closings:
        se = disk(tam)
        temp = closing_by_reconstruction(image, se)
        cbr[:, :, it] = temp[:, :]
        temp = opening_by_reconstruction(image, se)
        obr[:, :, it] = temp[:, :]
        tam += se_size_increment
        it += 1

    mp = np.zeros(shape=(x, y, (num_openings_closings*2)+1))
    cont = num_openings_closings - 1
    for i in range(num_openings_closings):
        mp[:, :, i] = cbr[:, :, cont]
        cont = cont - 1

    mp[:, :, num_openings_closings] = image[:, :]

    cont = 0
    for i in range(num_openings_closings+1, num_openings_closings*2+1):
        mp[:, :, i] = obr[:, :, cont]
        cont += 1

    return mp


def build_emp(base_image, se_size=4, se_size_increment=2, num_openings_closings=4):
    """
        Build the extended morphological profiles for a given set of images.

        Parameters:
            base_image: 3d matrix, each 'channel' is considered for applying the morphological profile. It is the spectral information part of the EMP.
            se_size: int, initial size of the structuring element (or kernel). Structuring Element used: disk
            se_size_increment: int, structuring element increment step
            num_openings_closings: int, number of openings and closings by reconstruction to perform.
        Returns:
            emp: 3d matrix with both spectral (from the base_image) and spatial information
    """
    base_image_channels, base_image_rows, base_image_columns = base_image.shape
    se_size = se_size
    se_size_increment = se_size_increment
    num_openings_closings = num_openings_closings
    morphological_profile_size = (num_openings_closings * 2) + 1
    emp_size = morphological_profile_size * base_image_channels
    emp = np.zeros(
        shape=(emp_size, base_image_rows, base_image_columns))

    cont = 0
    for i in range(base_image_channels):
        # build MPs
        mp_temp = build_morphological_profiles(
            base_image[i, :, :], se_size, se_size_increment, num_openings_closings)

        aux = morphological_profile_size * (i+1)

        # build the EMP
        cont_aux = 0
        for k in range(cont, aux):
            emp[k, :, :] = mp_temp[:, :, cont_aux]
            cont_aux += 1

        cont = morphological_profile_size * (i+1)

    return emp

class ExtendedMorphologicalProfiles(object):
    """
     by https://github.com/andreybicalho/ExtendedMorphologicalProfiles
    """
    def __init__(self, data: Optional[DataObject], num_openings_closings: int = 4):
        self.num_openings_closings = num_openings_closings

    def emp_transform(self, item):

        emp_image = build_emp(base_image=item.cpu().numpy(), num_openings_closings=self.num_openings_closings)

        return torch.from_numpy(emp_image).float()

    def __call__(self, batch):
        batch_ = []

        for item in batch:
            batch_.append(self.emp_transform(item))
        return torch.stack(batch_)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    d = get_data('fruit/papaya/ripeness/SPECIM_FX10', augmentations=[], data_set_root='/data2/datasets/deephs_benchmark')
    # d = get_data('remote_sensing/indian_pines/0.3', augmentations=[])

    emp = ExtendedMorphologicalProfiles(data=d, num_openings_closings=4)

    test_item = d.datasets.test[0][0]
    print(test_item.shape)
    plt.imshow(torch.squeeze(test_item.mean(0)))
    plt.savefig('/tmp/test.png')

    test_item_transformed = emp.emp_transform(test_item)
    print(test_item_transformed.shape)
    plt.imshow(torch.squeeze(test_item_transformed[500]))
    plt.savefig('/tmp/test_emp_comp.png')
    plt.imshow(test_item_transformed.mean(0))
    plt.savefig('/tmp/test_emp_mean.png')
