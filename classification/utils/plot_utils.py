import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.misc import toimage
from PIL import ImageFilter


def plot_(x, y, acc, save_path, value, pred_mean=None, pred_var=None, mean_entropy=None):
    img_w = x.shape[1]
    num_rows = int(np.sqrt(y.shape[0]))
    if not y.shape[0] % num_rows:
        num_cols = int(y.shape[0] / num_rows)
    else:
        num_cols = 1 + int(y.shape[0] / num_rows)
    # index of misclassified samples
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols)
    fig.set_size_inches(w=num_cols * 2.5, h=num_rows * 2.5)
    plt.subplots_adjust(left=0.1, bottom=0.13, right=0.9, top=0.9, wspace=0.5, hspace=0.5)
    for num_row in range(num_rows):
        for num_col in range(num_cols):
            ax = axes[num_row, num_col]
            indx = num_col + num_row * num_cols
            if pred_var is not None:
                color = 'g' if pred_mean[indx].argmax() == y[indx].argmax() else 'r'
            else:
                color = 'g' if pred_mean[indx].argmax() == y[indx].argmax() else 'r'
            ax.imshow(x[indx].reshape(img_w, img_w, -1), cmap='gray')
            plt.setp(axes[num_row, num_col].get_xticklabels(), visible=False)
            plt.setp(axes[num_row, num_col].get_yticklabels(), visible=False)
            if pred_var is not None:
                ax.set_title('T:{0}, P:{1}, {2:3.2f}({3:3.2f})'.format(y[indx].argmax(),
                                                                       pred_mean[indx].argmax(),
                                                                       pred_mean[indx].max(),
                                                                       pred_var[indx]),
                             color=color)
                fig.suptitle('radius={0}, accuracy={1:.02%}, mean entropy: {2:3.2f}'
                             .format(value, acc, mean_entropy))
            else:
                ax.set_title('T:{0}, P:{1}, (prob={2:3.2f})'.format(y[indx].argmax(),
                                                                    pred_mean[indx].argmax(),
                                                                    pred_mean[indx].max()),
                             color=color)
                fig.suptitle('radius={0}, accuracy={1:.02%}'.format(value, acc))
    fig.savefig(save_path+'.pdf')
    fig.savefig(save_path+'.svg')


def add_augment(batch, mean=0, var=0.1, amount=0.01, angle=0, radius=0, mode='pepper'):
    """Adding noise to a batch of images.
    :param batch: batch of images of size (#image, img_h, img_w, #channels)
    :param mean: mean of the Gaussian noise
    :param var: variance of the Gaussian noise
    :param amount: amount of the noise added for 'pepper' and 's&p'
    :param angle: rotation angle
    :param radius: Blurring filter radius
    :param mode: Noise type.
    """
    original_size = batch.shape
    batch = np.squeeze(batch)
    batch_noisy = np.zeros(batch.shape)
    for ii in range(batch.shape[0]):
        image = np.squeeze(batch[ii])
        if mode == 'gaussian':
            gauss = np.random.normal(mean, var, image.shape)
            image = image + gauss
        elif mode == 'pepper':
            num_pepper = np.ceil(amount * image.size)
            coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
            image[coords] = 0
        elif mode == "s&p":
            s_vs_p = 0.5
            # Salt mode
            num_salt = np.ceil(amount * image.size * s_vs_p)
            coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
            image[coords] = 1
            # Pepper mode
            num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
            coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
            image[coords] = 0
        elif mode == 'rotate':
            image = scipy.ndimage.interpolation.rotate(image, angle, mode='nearest', reshape=False)
        elif mode == 'blur':
            img = toimage(image)
            blur_image = img.filter(ImageFilter.GaussianBlur(radius=radius))
            image = np.array(blur_image)
        batch_noisy[ii] = image
    return batch_noisy.reshape(original_size)
