import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_string('mode', default='train', help='train or test')
flags.DEFINE_string('model', default='fcnet', help='fcnet or lenet')
flags.DEFINE_boolean('bayes', default=False, help='Whether to use Bayesian network '
                                                  '(MC simulations on CNN with dropout or dropconnect techniques)')
flags.DEFINE_integer('max_epoch', 2000, 'maximum number of training epochs')
flags.DEFINE_integer('monte_carlo_simulations', default=100, help='The number of monte carlo simulation runs')
flags.DEFINE_integer('reload_step', default=0, help='Reload step to continue training')

# Training logs
flags.DEFINE_integer('SUMMARY_FREQ', default=100, help='Number of step to save summary')
flags.DEFINE_integer('VAL_FREQ', 500, 'Number of step to evaluate the network on Validation data')
flags.DEFINE_float('init_lr', default=1e-3, help='Initial learning rate')
flags.DEFINE_float('lr_min', default=1e-5, help='Minimum learning rate')

# Hyper-parameters
flags.DEFINE_integer('batch_size', default=64, help='training batch size')
flags.DEFINE_integer('val_batch_size', default=100, help='validation batch size')
flags.DEFINE_float('lmbda', default=1e-4, help='L2 regularization coefficient')
flags.DEFINE_float('keep_prob', default=0.5, help='keep prob of the dropout')
flags.DEFINE_boolean('use_reg', default=True, help='Use L2 regularization on weights')

# data
flags.DEFINE_string('data', default='cifar', help='mnist or cifar')
flags.DEFINE_boolean('data_augment', default=True, help='whether to apply data augmentation or not')
flags.DEFINE_integer('max_angle', default=15, help='maximum rotation angle')
flags.DEFINE_integer('height', default=32, help='input image height size')
flags.DEFINE_integer('width', default=32, help='input image width size')
flags.DEFINE_integer('channel', default=3, help='input image channel size')
flags.DEFINE_integer('num_cls', default=10, help='Number of output classes')

# Directories
flags.DEFINE_string('run_name', default='dropout', help='Run name (none, dropout, or dropconnect)')
flags.DEFINE_string('logdir', default='./Results/log_dir/', help='Logs directory')
flags.DEFINE_string('modeldir', default='./Results/model_dir/', help='Model directory')
flags.DEFINE_string('imagedir', default='./Results/image_dir/', help='Directory to save sample predictions')
flags.DEFINE_string('model_name', default='model', help='Model file name')

args = tf.app.flags.FLAGS
