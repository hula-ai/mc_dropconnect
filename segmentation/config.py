import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_string('mode', default='train', help='train or test')
flags.DEFINE_boolean('bayes', default=False, help='Whether to use Bayesian network or not')
flags.DEFINE_integer('monte_carlo_simulations', default=50, help='The number of monte carlo simulation runs')
flags.DEFINE_integer('reload_step', default=0, help='Reload step to continue training')

# Data
flags.DEFINE_string('path_to_input_images_train', './data_preparation/CamVid/train/*.png', 'path to train input images')
flags.DEFINE_string('path_to_output_labels_train', './data_preparation/CamVid/trainannot/*.png', 'path to train output labels')
flags.DEFINE_string('path_to_input_images_val', './data_preparation/CamVid/val/*.png', 'path to val input images')
flags.DEFINE_string('path_to_output_labels_val', './data_preparation/CamVid/valannot/*.png', 'path to val output labels')
flags.DEFINE_string('path_to_input_images_test', './data_preparation/CamVid/test/*.png', 'path to test input images')
flags.DEFINE_string('path_to_output_labels_test', './data_preparation/CamVid/testannot/*.png', 'path to test output labels')

# Training logs
flags.DEFINE_integer('SUMMARY_FREQ', 100, 'Number of step to save summary')
flags.DEFINE_integer('VAL_FREQ', 500, 'Number of step to evaluate the network on Validation data')
flags.DEFINE_float('init_lr', 1e-3, 'Initial learning rate')
flags.DEFINE_float('lr_min', 1e-5, 'Minimum learning rate')

# Hyper-parameters
flags.DEFINE_boolean('weighted_loss', True, 'Whether to use weighted cross-entropy or not')
flags.DEFINE_string('loss_type', 'cross-entropy', 'cross-entropy or dice')
flags.DEFINE_boolean('use_reg', True, 'Use L2 regularization on weights')
flags.DEFINE_float('lmbda', 1e-4, 'L2 regularization coefficient')
flags.DEFINE_integer('batch_size', 2, 'training batch size')
flags.DEFINE_integer('val_batch_size', 2, 'training batch size')

# data
flags.DEFINE_string('data', 'ct', 'Training data name; ct or camvid')
flags.DEFINE_integer('dim', 2, 'input and network dimension; either 2 or 3')
flags.DEFINE_string('train_data_dir', '/data_preparation/our_data/4_correctMask_normalized/train/', 'Training data')
flags.DEFINE_string('valid_data_dir', '/data_preparation/our_data/4_correctMask_normalized/test/', 'Validation data ')
flags.DEFINE_string('test_data_dir', '/data_preparation/our_data/4_correctMask_normalized/test/', 'Test data')
flags.DEFINE_boolean('random_crop', True, 'Crops the input and output randomly during training time only')
flags.DEFINE_list('crop_size', [256, 256, 32], 'crop sizes')
flags.DEFINE_boolean('data_augment', True, 'Adds augmentation to data')
flags.DEFINE_integer('max_angle', 40, 'Maximum rotation angle along each axis; when applying augmentation')
flags.DEFINE_integer('height', 512, 'Original image (and Network if random_crop is off) height size')
flags.DEFINE_integer('width', 512, 'Original image (and Network if random_crop is off) width size')
flags.DEFINE_integer('channel', 1, 'Original image channel size')
flags.DEFINE_integer('depth', 32, 'Network depth size during training (if random_crop is off)')
flags.DEFINE_integer('Dcut_size', 32, 'Depth of the validation slices')

# Directories
flags.DEFINE_string('run_name', 'run_weighted_dropcon', 'Run name')
flags.DEFINE_string('logdir', './Results/log_dir/', 'Logs directory')
flags.DEFINE_string('modeldir', './Results/model_dir/', 'Model directory')
flags.DEFINE_string('imagedir', './Results/image_dir/', 'Directory to save sample predictions')
flags.DEFINE_string('model_name', 'model', 'Model file name')

# network architecture
flags.DEFINE_integer('num_cls', 6, 'Number of output classes')
flags.DEFINE_list('label_name', ['background', 'liver', 'spleen', 'kidney', 'bone', 'vessel'], 'class names')
# flags.DEFINE_list('label_name', ['sky', 'building', 'pole', 'road', 'pavement', 'tree', 'signsymbol'
#                                  , 'fence', 'car', 'pedestrian', 'bicyclist', 'unlabeled'], 'class names')
flags.DEFINE_boolean('use_BN', True, 'Adds Batch-Normalization to all convolutional layers')
flags.DEFINE_integer('start_channel_num', 16, 'start number of outputs for the first conv layer')
flags.DEFINE_integer('filter_size', 3, 'Filter size for the conv and deconv layers')
flags.DEFINE_integer('pool_filter_size', 2, 'Filter size for pooling layers')
flags.DEFINE_float('keep_prob', 0.5, 'Probability of keeping a unit in drop-out')

#
flags.DEFINE_float('theta_down', 0.3, 'transition down layers factor')
flags.DEFINE_float('theta_up', 0.3, 'transition up layers factor')
flags.DEFINE_integer('growth_rate', 32, 'Growth rate of the DenseNet')

# camvid
flags.DEFINE_string('img_prefix', '.', '')
flags.DEFINE_string('label_prefix', '.', '')

flags.DEFINE_string('train_file', 'SegNet/CamVid/train.txt', 'path to train text file')
flags.DEFINE_string('val_file', 'SegNet/CamVid/val.txt', 'path to validation text file')
flags.DEFINE_string('test_file', 'SegNet/CamVid/test.txt', 'path to test text file')

args = tf.app.flags.FLAGS