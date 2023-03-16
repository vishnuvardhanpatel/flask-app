import matplotlib.pyplot as plt
plt.switch_backend('agg')
from flask import Flask, render_template, request
import pathlib as P
import denoising
import tensorflow as tf
import numpy as np
from math import log10, sqrt

path = P.Path.cwd()
input_dir = f'{path}/input'
output_dir = f'{path}/output'
input_dir = f'{path}/input'

app = Flask(__name__)

def make_directory(f1_format, f2_format):
    if not P.Path(output_dir).exists():
        P.Path(output_dir).mkdir()
        P.Path(f'{output_dir}/{f1_format}_to_{f2_format}').mkdir()
    elif P.Path(f'{output_dir}/{f1_format}_to_{f2_format}').exists():
        pass
    else:
        P.Path(f'{output_dir}/{f1_format}_to_{f2_format}').mkdir()

@app.route('/',methods=['GET','POST'])
def Home():
    return render_template('index.html')

@app.route('/denoise/pngtopng', methods=['GET','POST'])
def png_to_png_execution():

    conv_path = 'png_to_png'
    input_file_format = 'png'
    output_file_format = 'png'

    inp_ds, adl_results = master_func(conv_path, input_file_format, output_file_format)

    # visualization functions
    display(-1, inp_ds, adl_results, f'{output_dir}/{conv_path}',output_file_format)
    psnr_val = PSNR(f'{output_dir}/{conv_path}',output_file_format)
    display_comparison(-1, inp_ds, adl_results,
                       f'{output_dir}/{conv_path}', output_file_format, psnr_val)

    msg = f"{input_file_format} image has been denoised & saved in directory: output/{conv_path}"
    # # return json instead of html
    return render_template('result.html', msg=msg)

@app.route('/denoise/dcmtopng', methods=['GET','POST'])
def dcm_to_png_execution():

    conv_path = 'dcm_to_png'
    input_file_format = 'dcm'
    output_file_format = 'png'

    inp_ds, adl_results = master_func(conv_path, input_file_format, output_file_format)
    # visualization functions
    display(-1, inp_ds, adl_results, f'{output_dir}/{conv_path}', output_file_format)
    psnr_val = PSNR(f'{output_dir}/{conv_path}', output_file_format)
    display_comparison(-1, inp_ds, adl_results,
                       f'{output_dir}/{conv_path}', output_file_format, psnr_val)

    msg = f"{input_file_format} image has been denoised & saved in directory: output/{conv_path}"
    return render_template('result.html',msg=msg)

@app.route('/denoise/dcmtojpeg', methods=['GET','POST'])
def dcm_to_jpeg_execution():

    conv_path = 'dcm_to_jpeg'
    input_file_format = 'dcm'
    output_file_format = 'jpeg'

    inp_ds, adl_results = master_func(conv_path, input_file_format, output_file_format)
    # visualization functions
    display(-1, inp_ds, adl_results, f'{output_dir}/{conv_path}', output_file_format)
    psnr_val = PSNR(f'{output_dir}/{conv_path}', output_file_format)
    display_comparison(-1, inp_ds, adl_results,
                       f'{output_dir}/{conv_path}', output_file_format, psnr_val)

    msg = f"{input_file_format} image has been denoised & saved in directory: output/{conv_path}"
    return render_template('result.html', msg=msg)


def master_func(conv_path, input_file_format, output_file_format):

    make_directory(input_file_format, output_file_format)
    obj1 = denoising.Denoiser()
    obj1.run_model()
    inp_ds, adl_results = obj1.run_colab(input_file_format, conv_path)

    return inp_ds, adl_results

def display( img_id, inp_ds, adl_results, output_path, output_file_format):

    # # saving noisy image
    plt.imshow(inp_ds[img_id])
    plt.savefig(f'{output_path}/noisy_image_.{output_file_format}')
    plt.close()
    # # saving denoised image
    plt.imshow(adl_results[img_id])
    plt.savefig(f'{output_path}/denoised_.{output_file_format}')
    plt.close()

def PSNR(img_path, output_file_format):

    image = tf.io.read_file(
        f'{img_path}\\noisy_image_.{output_file_format}')
    img1 = tf.io.decode_jpeg(image, channels=3)
    img1 = tf.image.convert_image_dtype(img1, tf.float32)
    image = tf.io.read_file(
        f'{img_path}\\denoised_.{output_file_format}')
    img2 = tf.io.decode_jpeg(image, channels=3)
    img2 = tf.image.convert_image_dtype(img2, tf.float32)

    mse = np.mean((img1 - img2) ** 2)
    if (mse == 0):  # MSE is zero means no noise is present in the signal .
        # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    # psnr_val = f"PSNR value between noisy and denoised image is {value} dB"

    return f"PSNR value between noisy and denoised image is {psnr} dB"

def display_comparison(img_id, inp_ds, adl_results, output_path, output_file_format, psnr_val):
    # # saving comparison image
    num_figs = 2
    fontsize = 15

    fig = plt.figure(figsize=(num_figs * 6, 8))

    ax2 = fig.add_subplot(1, num_figs, 1)
    plt.title(f'Noisy Image', fontsize=fontsize)
    ax2.axis('off')
    ax2.imshow(np.clip(inp_ds[img_id], 0, 1))

    ax3 = fig.add_subplot(1, num_figs, 2)
    plt.title('ADL output', fontsize=fontsize)
    ax3.axis('off')
    ax3.imshow(adl_results[img_id])
    fig.suptitle(psnr_val, fontsize=20)

    plt.savefig(f'{output_path}/Comparison_.{output_file_format}')
    plt.close()

if __name__ == "__main__":
    app.run(debug=True)