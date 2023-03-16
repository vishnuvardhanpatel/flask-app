import pathlib as P
import tensorflow as tf
import numpy as np
import pydicom
# import tensorflow as tf
# from math import log10, sqrt

path = P.Path.cwd()
output_dir = f'{path}/output'
input_dir = f'{path}/input'
path = P.Path.cwd()

def png_to_png():
    '''
        1. creates output/png_to_png directory
        2. Fetches png image from input/png and decodes the image as numpy array
    '''
    if not P.Path(output_dir).exists():
        P.Path(output_dir).mkdir()
        P.Path(f'{output_dir}/png_to_png').mkdir()
    elif not P.Path(f'{output_dir}/png_to_png').exists():
        P.Path(f'{output_dir}/png_to_png').mkdir()
    else:
        pass

    image_name = str(list(P.Path(f'{path}/input/png').glob('*.png'))[0])
    image_name = image_name.split('\\')[-1]

    file_path = f'{path}/input/png/{image_name}'
    file = tf.io.read_file(file_path)
    img = tf.image.decode_png(file, dtype=tf.uint8)

    return img

def dcm_to_png():
    '''
        1. creates output/dcm_to_png directory
        2. Fetches dcm image from input/dcm and decodes the image as numpy array
    '''
    if not P.Path(output_dir).exists():
        P.Path(output_dir).mkdir()
        P.Path(f'{output_dir}/dcm_to_png').mkdir()
    elif not P.Path(f'{output_dir}/dcm_to_png').exists():
        P.Path(f'{output_dir}/dcm_to_png').mkdir()
    else:
        pass

    image_name = str(list(P.Path(f'{path}/input/dcm').glob('*.dcm'))[0])
    image_name = image_name.split('\\')[-1]
    img = dicom_decoder(image_name)

    return img

def dcm_to_jpeg():
    '''
        1. creates output/dcm_to_jpeg directory
        2. Fetches dcm image from input/dcm and decodes the image as numpy array
    '''
    if not P.Path(output_dir).exists():
        P.Path(output_dir).mkdir()
        P.Path(f'{output_dir}/dcm_to_jpeg').mkdir()
    elif not P.Path(f'{output_dir}/dcm_to_jpeg').exists():
        P.Path(f'{output_dir}/dcm_to_jpeg').mkdir()
    else:
        pass

    image_name = str(list(P.Path(f'{path}/input/dcm').glob('*.dcm'))[0])
    image_name = image_name.split('\\')[-1]
    img = dicom_decoder(image_name)

    return img

def dicom_decoder(image_name):
    '''
        Fetches dcm image and decodes the image as numpy array
    '''
    ds = pydicom.dcmread(f'{path}/input/dcm/{image_name}')
    # Convert to float to avoid overflow or underflow losses.
    image_2d = ds.pixel_array.astype(float)
    # Rescaling grey scale between 0-255
    image_2d_scaled = (np.maximum(image_2d, 0) / image_2d.max()) * 256
    # Convert to uint
    image_2d_scaled = np.uint8(image_2d_scaled)
    stacked_img = np.stack((image_2d_scaled,) * 3, axis=-1)
    img = tf.convert_to_tensor(stacked_img)

    return img

