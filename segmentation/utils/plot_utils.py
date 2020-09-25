import numpy as np
import matplotlib.pyplot as plt
import h5py
from matplotlib import gridspec
import os


def create_pascal_label_colormap():
    """Creates a label colormap used in PASCAL VOC segmentation benchmark.

  Returns:
    A Colormap for visualizing segmentation results.
  """
    colormap = np.zeros((256, 3), dtype=int)
    ind = np.arange(256, dtype=int)

    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= ((ind >> channel) & 1) << shift
        ind >>= 3

    return colormap


def label_to_color_image(label):
    """Adds color defined by the dataset colormap to the label.

  Args:
    label: A 2D array with integer type, storing the segmentation label.

  Returns:
    result: A 2D array with floating type. The element of the array
      is the color indexed by the corresponding element in the input label
      to the PASCAL color map.

  Raises:
    ValueError: If label is not of rank 2 or its value is larger than color
      map maximum entry.
  """
    if label.ndim != 2:
        raise ValueError('Expect 2-D input label')

    colormap = create_pascal_label_colormap()

    if np.max(label) >= len(colormap):
        raise ValueError('label value too large.')

    return colormap[label]


def vis_segmentation(image, seg_map_gt, seg_map_pred, var_map_pred=None, cls_uncert=None, label_names=None, image_name=None):
    """Visualizes input image, segmentation map and overlay view."""
    FULL_LABEL_MAP = np.arange(len(label_names)).reshape(len(label_names), 1)
    FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)
    num_cls = len(label_names)

    if cls_uncert is not None:
        plt.figure(figsize=(20, 8))
        grid_spec = gridspec.GridSpec(2, num_cls)
    else:
        plt.figure(figsize=(20, 6))
        grid_spec = gridspec.GridSpec(1, 6, width_ratios=[6, 6, 6, 6, 6, 1])

    # plot input image
    ii = 0
    plt.subplot(grid_spec[ii])
    plt.imshow(np.squeeze(image), cmap='gray')
    plt.axis('off')
    plt.title('input image')

    # plot ground truth mask
    ii += 1
    plt.subplot(grid_spec[ii])
    seg_image = label_to_color_image(seg_map_gt.astype(np.int32)).astype(np.uint8)
    plt.imshow(seg_image)
    plt.axis('off')
    plt.title('ground truth map')

    ii += 1
    plt.subplot(grid_spec[ii])
    seg_image = label_to_color_image(seg_map_pred.astype(np.int32)).astype(np.uint8)
    plt.imshow(seg_image)
    plt.axis('off')
    plt.title('prediction map')

    ii += 1
    plt.subplot(grid_spec[ii])
    plt.imshow(np.squeeze(image), cmap='gray')
    plt.imshow(seg_image, alpha=0.4)
    plt.axis('off')
    plt.title('prediction overlay')
    if var_map_pred is not None:
        ii += 1
        ax = plt.subplot(grid_spec[ii])
        plt.subplot(grid_spec[ii])
        plt.imshow(var_map_pred, cmap='Greys')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.title('model uncertainty')

    ii += 1
    unique_labels = np.unique(np.concatenate((np.unique(seg_map_gt), np.unique(seg_map_pred)), 0)).astype(np.int32)
    ax = plt.subplot(grid_spec[ii])
    plt.imshow(FULL_COLOR_MAP[unique_labels].astype(np.uint8), interpolation='nearest')
    ax.yaxis.tick_right()
    plt.yticks(range(len(unique_labels)), label_names[unique_labels])
    plt.xticks([], [])
    ax.tick_params(width=0.0)
    plt.grid('off')

    if cls_uncert is not None:
        for i, name in enumerate(label_names):
            ax = plt.subplot(grid_spec[i + ii + 1])
            plt.subplot(grid_spec[i + ii + 1])
            plt.imshow(cls_uncert[:, :, i], cmap='Greys')
            # plt.axis('off')
            plt.title(name)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

    plt.savefig(image_name)


def plot_save_preds_3d(images, masks, mask_preds, var_preds=None, slice_numbers=None,
                       depth=None, path=None, label_names=None):
    if not os.path.exists(path):
        os.makedirs(path)
    if var_preds is None:
        for slice_num, image, mask, mask_pred in zip(slice_numbers, images, masks, mask_preds):
            img_name = os.path.join(path, str(slice_num) + '_' + str(depth) + '.png')
            vis_segmentation(image, mask, mask_pred, label_names=label_names, image_name=img_name)
    else:
        for slice_num, image, mask, mask_pred, var_pred in zip(slice_numbers, images, masks, mask_preds, var_preds):
            img_name = os.path.join(path, str(slice_num) + '_' + str(depth) + '.png')
            vis_segmentation(image, mask, mask_pred, var_pred, label_names, img_name)


def plot_save_preds_2d(images, masks, mask_preds, var_preds=None, cls_unc=None, path=None, label_names=None):
    slice_numbers = list(range(masks.shape[0]))
    if not os.path.exists(path):
        os.makedirs(path)
    if var_preds is None:
        for slice_num, image, mask, mask_pred in zip(slice_numbers, images, masks, mask_preds):
            img_name = os.path.join(path, str(slice_num) + '.png')
            vis_segmentation(image, mask, mask_pred, label_names=label_names, image_name=img_name)
    else:
        if cls_unc is None:
            for slice_num, image, mask, mask_pred, var_pred in zip(slice_numbers, images, masks, mask_preds, var_preds):
                img_name = os.path.join(path, str(slice_num) + '.png')
                vis_segmentation(image, mask, mask_pred, var_pred, label_names=label_names, image_name=img_name)
        else:
            for slice_num, image, mask, mask_pred, var_pred, cls_un in zip(slice_numbers, images, masks, mask_preds, var_preds, cls_unc):
                img_name = os.path.join(path, str(slice_num) + '.png')
                vis_segmentation(image, mask, mask_pred, var_pred, cls_un, label_names, img_name)


if __name__ == '__main__':
    LABEL_NAMES = np.asarray(['background', 'liver', 'spleen', 'kidney', 'bone', 'vessel'])
    File_path = '/home/cougarnet.uh.edu/amobiny/Desktop/CT_Semantic_Segmentation/data_preparation/' \
                'our_data/4_correctMask_normalized/new_train/PV_anon_1579_5_232_ARLS1.h5'
    h5f = h5py.File(File_path, 'r')
    x = np.squeeze(h5f['x'][:])
    x_norm = np.squeeze(h5f['x_norm'][:])
    y = np.squeeze(h5f['y'][:])
    h5f.close()
    image = x_norm[:, :, 10]
    true_mask = y[:, :, 10]
    vis_segmentation(image, true_mask, true_mask, label_names=LABEL_NAMES, image_name='test.png')

print()
