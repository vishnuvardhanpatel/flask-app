import pathlib as P
import tensorflow as tf
import numpy as np

config = {
    "H": -1,
    "W": -1,
    "num_channels": 3,
    "batch_size_per_gpu": 1,
    "adding_noise": True,
    "adding_blur": False,  # True if the model is uploaded
    "adding_compression": False,  # True if the model is uploaded
    "test_stdVec": [5., 10., 20.],  # noise level (sigma/255)
    "test_blurVec": [0],
    "test_compresVec": [1.],
    "localhost": None,
    "img_types": ["png", "jpg", "jpeg", "bmp", "dcm"],
    "num_sel_imgs": -1
}

import DataLoader_colab
# exec(open('DataLoader_colab.py').read())
# %run '/content/ADL/TensorFlow/util/DataLoader_colab.py

# # loading model

class Denoiser():

    def __init__(self):
        self.path = P.Path.cwd()
        self.input_dir = f'{self.path}/input'
        self.output_dir = f'{self.path}/output'
        self.gt_ds, self.inp_ds, self.adl_results, self.noise_level = [], [], [], []

    def run_model(self):
        self.model = tf.keras.models.load_model(
            r'C:\\Users\\PranavD2\Downloads\\test\\ADL\\TensorFlow\\pretrained_models\\RGB_model_WGN\\checkpoint-36\\',
            compile=False)
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    def run_colab(self, file_format, conv_func):
        test_DSs = DataLoader_colab.DataLoader(config=config, test_ds_dir=self.input_dir,
                                               file_format=file_format,
                                               conv_func = conv_func)()

        for ds_name, DSs in test_DSs.items():
            print(f"dataset: {ds_name}...")

            for distortion_name, DS in DSs.items():
                print(f"\tdistortion type: {distortion_name}...")
                sigma = int(float(distortion_name.split('_wgn_')[-1]))
                self.noise_level.append(sigma)

                for inp, gt, img_name in DS.batch(1):
                    y_hat, _, _ = self.model.predict(inp)

                    self.gt_ds.append(np.squeeze(gt.numpy()).astype(np.float32))
                    self.inp_ds.append(np.squeeze(inp.numpy()).astype(np.float32))
                    self.adl_results.append(np.squeeze(tf.identity(y_hat).numpy()).astype(np.float32))

        return self.inp_ds, self.adl_results
