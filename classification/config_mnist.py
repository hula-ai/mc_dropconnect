import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_string('mode', 'test', 'train or test')
flags.DEFINE_string('model', 'lenet', 'lenet')
flags.DEFINE_boolean('bayes', True, 'Whether to use Bayesian network or not')
flags.DEFINE_integer('max_epoch', 2000, 'maximum number of training epochs')
flags.DEFINE_integer('monte_carlo_simulations', 1000, 'The number of monte carlo simulation runs')
flags.DEFINE_integer('reload_step', 27450, 'Reload step to continue training')

# Training logs
flags.DEFINE_integer('max_step', 300000, '# of step for training')
flags.DEFINE_integer('SUMMARY_FREQ', 100, 'Number of step to save summary')
flags.DEFINE_integer('VAL_FREQ', 500, 'Number of step to evaluate the network on Validation data')
flags.DEFINE_float('init_lr', 1e-3, 'Initial learning rate')
flags.DEFINE_float('lr_min', 1e-5, 'Minimum learning rate')

# Hyper-parameters
flags.DEFINE_integer('batch_size', 100, 'training batch size')
flags.DEFINE_integer('val_batch_size', 100, 'validation batch size')
flags.DEFINE_float('lmbda', 1e-4, 'L2 regularization coefficient')
flags.DEFINE_float('keep_prob', 0.5, 'keep prob of the dropout')
flags.DEFINE_boolean('use_reg', True, 'Use L2 regularization on weights')

# data
flags.DEFINE_string('data', 'mnist', 'mnist or camvid')
flags.DEFINE_boolean('data_augment', True, 'whether to apply data augmentation or not')
flags.DEFINE_integer('max_angle', 15, 'maximum rotation angle')
flags.DEFINE_integer('height', 28, 'input image height size')
flags.DEFINE_integer('width', 28, 'input image width size')
flags.DEFINE_integer('channel', 1, 'input image channel size')
flags.DEFINE_integer('num_cls', 10, 'Number of output classes')

# Directories
flags.DEFINE_string('run_name', 'new_dropout', 'Run name')
flags.DEFINE_string('logdir', './Results/log_dir/', 'Logs directory')
flags.DEFINE_string('modeldir', './Results/model_dir/', 'Model directory')
flags.DEFINE_string('imagedir', './Results/image_dir/', 'Directory to save sample predictions')
flags.DEFINE_string('model_name', 'model', 'Model file name')

args = tf.app.flags.FLAGS