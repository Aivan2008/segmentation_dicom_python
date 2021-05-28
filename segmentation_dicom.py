import numpy as np # linear algebra
import pandas as pd # reading and processing of tables
import skimage
import os
from skimage.morphology import ball, disk, dilation, binary_erosion, remove_small_objects, erosion, closing, reconstruction, binary_closing
from skimage.measure import label,regionprops, perimeter
from skimage.morphology import binary_dilation, binary_opening
from skimage.filters import roberts, sobel
from skimage import measure, feature
from skimage.segmentation import clear_border
from skimage.util import montage
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pydicom
import scipy.misc
import cv2
import nibabel as nb
from numba import njit


thresh = -1800
affine = None
min_z = 10000
CT_OFFSET = 1024
ZERO_VALUE = -2000

def inverse_segmentation(img):
    """
    Функция инверсии разметки, закрашивает все, что было неразмечено и удаляет все, что" конектится с краями
    """
    new_img = (img == 0.) * 1.
    cleared = clear_border(new_img)
    return cleared

def get_series_name_and_len(folder_name):
    volume_dict = dict()
    
    for filename in os.listdir(folder_name):
        dicom = pydicom.read_file(os.path.join(folder_name, filename))
        
        volume_name = dicom.SeriesDescription
        if volume_name in volume_dict.keys():
            volume_dict[volume_name] += 1
        else:
            volume_dict[volume_name] = 1
    return volume_dict

def read_dicom_array(in_path, volume_name):
    # type: (str) -> Tuple[int, np.ndarray]
    lung_dicom = pydicom.read_file(in_path)
    
    if lung_dicom.SeriesDescription != volume_name:
        return None
    
    try:
        im_pos_pat = lung_dicom.ImagePositionPatient
    except:
        try:
            im_pos_pat = [float(x) for x in lung_dicom[('0020', '0032')]]
        except:
            return None
    z_location = im_pos_pat[2]    
        
    slice_array = lung_dicom.pixel_array
    slice_array[slice_array == ZERO_VALUE] = 0
    
    global affine, min_z
    affine = np.asarray([[lung_dicom.PixelSpacing[0], 0, 0, im_pos_pat[0]], 
                         [0, lung_dicom.PixelSpacing[1], 0, im_pos_pat[1]],
                         [0, 0, 0, 0],
                         [0, 0, 0, 1]])
    
    if z_location < min_z:
        min_z = z_location
        
    return int(lung_dicom.InstanceNumber), slice_array.astype(np.int16) - CT_OFFSET, z_location

def read_ct_scan(folder_name, volume_name):
        # Read the slices from the dicom file
        slices = []
        for filename in os.listdir(folder_name):
            slice = read_dicom_array(os.path.join(folder_name, filename), volume_name)
            if not slice is None:
                slices.append(slice)
        
        # Sort the dicom slices in their respective order
        s_slices = sorted(slices, key = lambda x: x[2])
        
        global affine
        affine[2,2] = s_slices[1][2] - s_slices[0][2] 
        
        # Get the pixel values for all the slices
        slices = np.stack([data for pos, data, z_location in s_slices])
        return slices
    
def plot_ct_scan(scan):
    """
    show all the slices the ct as a montage
    """
    f, ax1 = plt.subplots(1,1, figsize=(12, 12))
    ax1.imshow(montage(scan), cmap=plt.cm.bone) 
    ax1.axis('off')
    
def get_segmented_lungs(in_im, plot=False, threshold=0.26):
    im = in_im.copy() # don't change the input
    im = cv2.rotate(im, cv2.ROTATE_90_COUNTERCLOCKWISE)
    im = cv2.flip(im, 0)
    
    if threshold > 0 and threshold < 1:
        data = im

        min1 = data.min()
        min2 = data[data != min1].min()
        data[data == min1] = min2

        im = (data - data.min()) / (data.max() - data.min())
    '''
    This funtion segments the lungs from the given 2D slice.
    '''
    if plot == True:
        f, plots = plt.subplots(3, 3, figsize=(10, 10))
        plots = plots.flatten()
    '''
    Step 1: Convert into a binary image. 
    '''
    binary = im > threshold
    if plot == True:
        plots[0].axis('off')
        plots[0].imshow(binary, cmap=plt.cm.bone) 
        plots[0].set_title('First Threshold')
    '''
    Step 2: Remove the blobs connected to the border of the image.
    '''
    cleared = clear_border(binary)
    if plot == True:
        plots[1].axis('off')
        plots[1].imshow(cleared, cmap=plt.cm.bone) 
        plots[1].set_title('Remove Border')
    '''
    Step 3: Label the image.
    '''
    label_image = label(binary)
    if plot == True:
        plots[2].axis('off')
        plots[2].imshow(label_image, cmap=plt.cm.gist_earth)
        plots[2].set_title('Label Components')

    
    selem = disk(2)
    binary_palka = binary_dilation(label_image, selem)
    
    label_binary_palka = label(binary_palka)
    areas_palka = [r.area for r in regionprops(label_binary_palka)]
    areas_palka.sort()
    if len(areas_palka) > 1:
        for region in regionprops(label_binary_palka):
            if region.area < areas_palka[-1]:
                for coordinates in region.coords:                
                       label_binary_palka[coordinates[0], coordinates[1]] = 0
    label_binary_palka = binary_closing((label_binary_palka > 0) * 1.0, disk(2))
    
    binary_body = (label_binary_palka > 0) * 1.0
    binary_lung = inverse_segmentation(binary_body)

    label_all = (label_image > 0) * 1.0
    binary_palka = ((label_all - binary_body - binary_lung) > 0) * 2.0
    binary_palka[:, int(binary_palka.shape[0]/2):binary_palka.shape[0]] = 0
    
    '''
    Step 5: Erosion operation with a disk of radius 2. This operation is 
    seperate the lung nodules attached to the blood vessels.
    '''
    selem = disk(2)
    binary = binary_erosion(label_image, selem)
    if plot == True:
        plots[3].axis('off')
        plots[3].imshow(binary, cmap=plt.cm.bone)
        plots[3].set_title('Erosion')
    '''
    Step 4: Keep the labels with 1 largest area.
    '''
    label_binary = label(binary)
    areas = [r.area for r in regionprops(label_binary)]
    areas.sort()
    if len(areas) > 1:
        for region in regionprops(label_binary):
            if region.area < areas[-1]:
                for coordinates in region.coords:                
                       label_binary[coordinates[0], coordinates[1]] = 0
    binary = label_binary > 0
    if plot == True:
        plots[4].axis('off')
        plots[4].imshow(binary, cmap=plt.cm.bone) 
        plots[4].set_title('Keep Biggest 2')
    
    
    '''
    Step 5: Erosion operation with a disk of radius 2. This operation is 
    seperate the lung nodules attached to the blood vessels.
    '''
    selem = disk(2)
    binary_er = binary_erosion(binary, selem)
    if plot == True:
        plots[5].axis('off')
        plots[5].imshow(binary_er, cmap=plt.cm.bone)
        plots[5].set_title('Erosion')
    '''
    Step 6: Closure operation with a disk of radius 10. This operation is 
    to keep nodules attached to the lung wall.
    '''
    selem = disk(3)
    binary_er = binary_closing(binary, selem)
    if plot == True:
        plots[6].axis('off')
        plots[6].imshow(binary, cmap=plt.cm.bone) 
        plots[6].set_title('Close Image')
    binary = binary_dilation(binary, selem)
    '''
    Step 7: Fill in the small holes inside the binary mask of lungs.
    '''
    edges = roberts(binary)
    binary_1 = ndi.binary_fill_holes(edges)
    if plot == True:
        plots[7].axis('off')
        plots[7].imshow(binary_1, cmap=plt.cm.bone) 
        plots[7].set_title('Fill holes')
    '''
    Step 8: Superimpose the binary mask on the input image.
    '''
    get_high_vals = (binary_1 == 0)
    im[get_high_vals] = ZERO_VALUE # minimum value
    if plot == True:
        plots[8].axis('off')
        plots[8].imshow(im, cmap=plt.cm.bone) 
        plots[8].set_title('Binary Masked Input')
        
    return (binary_er > 0) * 1.0 + binary_palka

@njit
def find_staff(volume):
    for i in range(volume.shape[0]):
        for j in range(volume.shape[1]):
            for k in range(volume.shape[2]):
                if volume[i, j, k] == 2:
                    if volume[i-10: i+10, j-10: j+10, k-10: k+10].sum() / 2 < 1700:
                        volume[i, j, k] = 0
                    

def segmentation(path_to_dicom_folder, save_filename, volume_name, progress_bar = None):
    global tresh, affine, min_z
    thresh = -1700
    affine = None
    min_z = 10000
    
    SAVE_FILENAME_NII = save_filename
    ct_scan = read_ct_scan(path_to_dicom_folder, volume_name) 
    print('Scan Dimensions:',ct_scan.shape)
    affine[2, 3] = min_z

    # Умножаем элементы на -1, а саму картинку мы повернем на 270 градусов, уж не знаю зач, но надо
    affine[0, 0] *= -1
    affine[1, 1] *= -1
    affine[0, 3] *= -1
    affine[1, 3] *= -1
    print(affine)
    masks = []
    print(ct_scan.min())
    
    if ct_scan.min() == -1024:
        thresh = -700
        
    for i, sc in enumerate(ct_scan):
        mask = get_segmented_lungs(sc, False, thresh)
        masks.append(mask.reshape(mask.shape[0], mask.shape[1]))
        if progress_bar is not None:
            progress_bar.setValue(i / ct_scan.shape[0] * 60)
    if False:
        cv2.imshow('test', masks[50])
        cv2.waitKey()
        cv2.destroyAllWindows()
    result = np.moveaxis(np.asarray(masks), [0], [2])
    
    find_staff(result)
    if progress_bar is not None:
        progress_bar.setValue(i / ct_scan.shape[0] * 80)
    #find_staff(result)
    print(result.shape)
    new_image = nb.Nifti1Image(result.astype('float'), affine)
    nb.save(new_image, SAVE_FILENAME_NII)
    if progress_bar is not None:
        progress_bar.setValue(100)
    
if __name__=="__main__":
    save_filename = "C:\\Users\\User\\algorithms\\Lungs_db\\data_3\\Alekseev.nii.gz"
    path_to_dicom_folder = 'C:\\Users\\User\\algorithms\\Lungs_db\\data_3\\Alekseev\\synt_data'
    
    volume_dict = get_series_name_and_len(path_to_dicom_folder)
    
    print('Доступные объемы:')
    for i, volume_name in enumerate(volume_dict):
        print(i, volume_name, volume_dict[volume_name])
        
    print('Выберете объем (введите его номер):')
    num_vol = int(input())
    print(list(volume_dict.keys())[num_vol])
    volume_name = list(volume_dict.keys())[num_vol]
    segmentation(path_to_dicom_folder, save_filename, volume_name)