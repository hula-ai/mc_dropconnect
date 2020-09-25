import tensorflow as tf
from sklearn.metrics import roc_curve, precision_recall_curve
from tqdm import tqdm
from utils.loss_utils import cross_entropy
import os
import numpy as np

from utils.plot_utils import plot_, add_augment


class BaseModel(object):

    def __init__(self, sess, conf):
        self.sess = sess
        self.conf = conf
        self.input_shape = [None, self.conf.height, self.conf.width, self.conf.channel]
        self.output_shape = [None, self.conf.num_cls]
        self.create_placeholders()

    def create_placeholders(self):
        with tf.name_scope('Input'):
            self.inputs_pl = tf.placeholder(tf.float32, self.input_shape, name='input')
            self.labels_pl = tf.placeholder(tf.int64, self.output_shape, name='annotation')
            self.keep_prob_pl = tf.placeholder(tf.float32)

    def loss_func(self):
        with tf.name_scope('Loss'):
            self.y_prob = tf.nn.softmax(self.logits, axis=-1)
            with tf.name_scope('cross_entropy'):
                loss = cross_entropy(self.labels_pl, self.logits)
            with tf.name_scope('total'):
                if self.conf.use_reg:
                    with tf.name_scope('L2_loss'):
                        l2_loss = tf.reduce_sum(
                            self.conf.lmbda * tf.stack([tf.nn.l2_loss(v) for v in tf.get_collection('weights')]))
                        self.total_loss = loss + l2_loss
                else:
                    self.total_loss = loss
                self.mean_loss, self.mean_loss_op = tf.metrics.mean(self.total_loss)

    def accuracy_func(self):
        with tf.name_scope('Accuracy'):
            self.y_pred = tf.argmax(self.logits, axis=1, name='y_pred')
            self.y_prob = tf.nn.softmax(self.logits, axis=1)
            self.y_pred_ohe = tf.one_hot(self.y_pred, depth=self.conf.num_cls)
            correct_prediction = tf.equal(tf.argmax(self.labels_pl, axis=1), self.y_pred, name='correct_pred')
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy_op')
            self.mean_accuracy, self.mean_accuracy_op = tf.metrics.mean(accuracy)

    def configure_network(self):
        self.loss_func()
        self.accuracy_func()
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        learning_rate = tf.train.exponential_decay(self.conf.init_lr,
                                                   global_step,
                                                   decay_steps=2000,
                                                   decay_rate=0.97,
                                                   staircase=True)
        self.learning_rate = tf.maximum(learning_rate, self.conf.lr_min)
        with tf.name_scope('Optimizer'):
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.train_op = optimizer.minimize(self.total_loss, global_step=global_step)
        self.sess.run(tf.global_variables_initializer())
        trainable_vars = tf.trainable_variables()
        self.saver = tf.train.Saver(var_list=trainable_vars, max_to_keep=1000)
        self.train_writer = tf.summary.FileWriter(self.conf.logdir + self.conf.run_name + '/train/', self.sess.graph)
        self.valid_writer = tf.summary.FileWriter(self.conf.logdir + self.conf.run_name + '/valid/')
        self.configure_summary()
        print('*' * 50)
        print('Total number of trainable parameters: {}'.
              format(np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])))
        print('*' * 50)

    def configure_summary(self):
        summary_list = [tf.summary.scalar('learning_rate', self.learning_rate),
                        tf.summary.scalar('loss', self.mean_loss),
                        tf.summary.scalar('accuracy', self.mean_accuracy)]
        self.merged_summary = tf.summary.merge(summary_list)

    def save_summary(self, summary, step, mode):
        # print('----> Summarizing at step {}'.format(step))
        if mode == 'train':
            self.train_writer.add_summary(summary, step)
        elif mode == 'valid':
            self.valid_writer.add_summary(summary, step)
        self.sess.run(tf.local_variables_initializer())

    def train(self):
        self.sess.run(tf.local_variables_initializer())
        if self.conf.reload_step > 0:
            self.reload(self.conf.reload_step)
            print('----> Continue Training from step #{}'.format(self.conf.reload_step))
            self.best_validation_loss = input('Enter the approximate best validation loss you got last time')
            self.best_accuracy = input('Enter the approximate best validation accuracy (in range [0, 1])')
        else:
            self.best_validation_loss = 100
            self.best_accuracy = 0
            print('----> Start Training')
        if self.conf.data == 'mnist':
            from DataLoaders.mnist_loader import DataLoader
        elif self.conf.data == 'cifar':
            from DataLoaders.CIFARLoader import DataLoader
        else:
            print('wrong data name')
        self.data_reader = DataLoader(self.conf)
        self.data_reader.get_data(mode='train')
        self.data_reader.get_data(mode='valid')
        self.num_train_batch = self.data_reader.count_num_batch(self.conf.batch_size, mode='train')
        self.num_batch = self.data_reader.count_num_batch(self.conf.batch_size, mode='valid')
        for epoch in range(self.conf.max_epoch):
            self.data_reader.randomize()
            for train_step in range(self.num_train_batch):
                glob_step = epoch * self.num_train_batch + train_step
                start = train_step * self.conf.batch_size
                end = (train_step + 1) * self.conf.batch_size
                x_batch, y_batch = self.data_reader.next_batch(start, end, mode='train')
                feed_dict = {self.inputs_pl: x_batch, self.labels_pl: y_batch, self.keep_prob_pl: self.conf.keep_prob}
                if train_step % self.conf.SUMMARY_FREQ == 0 and train_step != 0:
                    _, _, _, summary = self.sess.run([self.train_op,
                                                      self.mean_loss_op,
                                                      self.mean_accuracy_op,
                                                      self.merged_summary], feed_dict=feed_dict)
                    loss, acc = self.sess.run([self.mean_loss, self.mean_accuracy])
                    self.save_summary(summary, glob_step + self.conf.reload_step, mode='train')
                    print('epoch {0}/{1}, step: {2:<6}, train_loss= {3:.4f}, train_acc={4:.01%}'.
                          format(epoch, self.conf.max_epoch, glob_step, loss, acc))
                else:
                    self.sess.run([self.train_op, self.mean_loss_op, self.mean_accuracy_op], feed_dict=feed_dict)
                if glob_step % self.conf.VAL_FREQ == 0 and glob_step != 0:
                    self.evaluate(train_step=glob_step, dataset='valid')

    def test(self, step_num, dataset='test'):
        self.sess.run(tf.local_variables_initializer())
        print('loading the model.......')
        self.reload(step_num)
        if self.conf.data == 'mnist':
            from DataLoaders.mnist_loader import DataLoader
        elif self.conf.data == 'mnist_bg':
            from DataLoaders.bg_mnist_loader import DataLoader
        elif self.conf.data == 'cifar':
            from DataLoaders.CIFARLoader import DataLoader
        else:
            print('wrong data name')
        self.data_reader = DataLoader(self.conf)
        self.data_reader.get_data(mode=dataset)
        self.num_batch = self.data_reader.count_num_batch(self.conf.batch_size, mode=dataset)

        print('-' * 25 + 'Test' + '-' * 25)
        if not self.conf.bayes:
            self.evaluate(dataset=dataset, train_step=step_num)
        else:
            self.MC_evaluate(dataset=dataset, train_step=step_num)

    def save(self, step):
        print('----> Saving the model at step #{0}'.format(step))
        checkpoint_path = os.path.join(self.conf.modeldir + self.conf.run_name, self.conf.model_name)
        self.saver.save(self.sess, checkpoint_path, global_step=step)

    def reload(self, step):
        checkpoint_path = os.path.join(self.conf.modeldir + self.conf.run_name, self.conf.model_name)
        model_path = checkpoint_path + '-' + str(step)
        if not os.path.exists(model_path + '.meta'):
            print('----> No such checkpoint found', model_path)
            return
        print('----> Restoring the model...')
        self.saver.restore(self.sess, model_path)
        print('----> Model successfully restored')

    def evaluate(self, dataset='valid', train_step=None):
        self.sess.run(tf.local_variables_initializer())
        for step in range(self.num_batch):
            start = self.conf.val_batch_size * step
            end = self.conf.val_batch_size * (step + 1)
            data_x, data_y = self.data_reader.next_batch(start=start, end=end, mode=dataset)
            feed_dict = {self.inputs_pl: data_x,
                         self.labels_pl: data_y,
                         self.keep_prob_pl: 1}
            self.sess.run([self.mean_loss_op, self.mean_accuracy_op], feed_dict=feed_dict)

        loss, acc = self.sess.run([self.mean_loss, self.mean_accuracy])
        if dataset == "valid":  # save the summaries and improved model in validation mode
            print('-' * 30)
            print('valid_loss = {0:.4f}, val_acc = {1:.01%}'.format(loss, acc))
            summary_valid = self.sess.run(self.merged_summary, feed_dict=feed_dict)
            self.save_summary(summary_valid, train_step, mode='valid')
            if loss < self.best_validation_loss:
                self.best_validation_loss = loss
                if acc > self.best_accuracy:
                    self.best_accuracy = acc
                    print('>>>>>>>> Both model validation loss and accuracy improved; saving the model......')
                else:
                    print('>>>>>>>> model validation loss improved; saving the model......')
                self.save(train_step)
            elif acc > self.best_accuracy:
                self.best_accuracy = acc
                print('>>>>>>>> model accuracy improved; saving the model......')
                self.save(train_step)
            print('-' * 30)
        elif dataset == 'test':
            print('test_loss = {0:.4f}, test_acc = {1:.02%}'.format(loss, acc))

    def MC_evaluate(self, dataset='test', train_step=None):
        num_rounds = 10
        self.sess.run(tf.local_variables_initializer())
        all_std, all_error = np.array([]), np.array([])
        mean_tpr, std_tpr = np.array([]), np.array([])
        mean_npv, std_npv = np.array([]), np.array([])
        mean_acc, std_acc = np.array([]), np.array([])

        all_sample = range(1, 100, 1)
        for mc_sample in tqdm(all_sample):
            round_error = np.array([])
            round_acc, round_npv, round_tpr = np.array([]), np.array([]), np.array([])
            for round_ in range(num_rounds):
                err = 0.
                mean_pred, var_pred = np.zeros_like(self.data_reader.y_test), np.zeros_like(self.data_reader.y_test[:, 0])

                for step in range(self.num_batch):
                    y_pred = np.zeros((mc_sample, self.conf.val_batch_size, self.conf.num_cls))
                    start = self.conf.val_batch_size * step
                    end = self.conf.val_batch_size * (step + 1)
                    data_x, data_y = self.data_reader.next_batch(start=start, end=end, mode='test')
                    # augment(data_y)
                    feed_dict = {self.inputs_pl: data_x,
                                 self.labels_pl: data_y,
                                 self.keep_prob_pl: self.conf.keep_prob}
                    for sample_id in range(mc_sample):
                        # save predictions from a sample pass
                        y_pred[sample_id] = self.sess.run(self.y_prob, feed_dict=feed_dict)

                    # average and variance over all passes
                    mean_pred[start:end] = y_pred.mean(axis=0)
                    # var_pred[start: end] = predictive_entropy(mean_pred[start:end])
                    var_pred[start: end] = mutual_info(mean_pred[start:end], y_pred)

                    # compute batch error
                    err += np.count_nonzero(np.not_equal(mean_pred[start:end].argmax(axis=1),
                                                         data_y.argmax(axis=1)))
                # accuracy = 1 - (err / mean_pred.shape[0])
                pred_error = err / mean_pred.shape[0]
                round_error = np.append(round_error, pred_error)
                tpr, npv, acc = compute_uncertainty_metrics(self.data_reader.y_test, mean_pred, var_pred,
                                                            desired_threshold=0.2)
                round_tpr = np.append(round_tpr, tpr)
                round_acc = np.append(round_acc, acc)
                round_npv = np.append(round_npv, npv)

            print('mc_sample = {0}, prediction error={1:.02%} +- {2:.4f}, TPR={3:.2f} +- {4:.4f}, '
                  'NPV={5:.2f} +- {6:.4f}, ACC={7:.2f} +- {8:.4f}'.format(mc_sample, np.mean(round_error),
                                                                          np.std(round_error),
                  np.mean(round_tpr), np.std(round_tpr), np.mean(round_npv),
                  np.std(round_npv), np.mean(round_acc), np.std(round_acc)))

            all_error = np.append(all_error, np.mean(round_error))
            all_std = np.append(all_std, np.std(round_error))

            mean_tpr = np.append(mean_tpr, np.mean(round_tpr))
            std_tpr = np.append(std_tpr, np.std(round_tpr))
            mean_npv = np.append(mean_npv, np.mean(round_npv))
            std_npv = np.append(std_npv, np.std(round_npv))
            mean_acc = np.append(mean_acc, np.mean(round_acc))
            std_acc = np.append(std_acc, np.std(round_acc))

    def MC_evaluate_plot(self):
        IMAGE_DIR = './classification/images/' + self.conf.run_name + '/'

        if not os.path.exists(IMAGE_DIR):
            os.makedirs(IMAGE_DIR)

        self.sess.run(tf.local_variables_initializer())
        err = 0.
        idx = np.array(range(100))
        angles = [0]
        for ii, angle in enumerate(angles):
            err = 0.
            mean_pred, var_pred = np.zeros_like(self.data_reader.y_test), np.zeros_like(self.data_reader.y_test[:, 0])
            for step in tqdm(range(self.num_batch)):
                y_pred = np.zeros((self.conf.monte_carlo_simulations, self.conf.val_batch_size, self.conf.num_cls))
                start = self.conf.val_batch_size * step
                end = self.conf.val_batch_size * (step + 1)
                data_x, data_y = self.data_reader.next_batch(start=start, end=end, mode='test')
                data_x_aug = add_augment(data_x, mode='rotate', angle=angle)
                feed_dict = {self.inputs_pl: data_x_aug,
                             self.labels_pl: data_y,
                             self.keep_prob_pl: self.conf.keep_prob}
                for sample_id in range(self.conf.monte_carlo_simulations):
                    # save predictions from a sample pass
                    y_pred[sample_id] = self.sess.run(self.y_prob, feed_dict=feed_dict)

                # average and variance over all passes
                mean_pred[start:end] = y_pred.mean(axis=0)
                # var_pred[start: end] = predictive_entropy(mean_pred[start:end])
                var_pred[start: end] = mutual_info(mean_pred[start:end], y_pred)

                # compute batch error
                err += np.count_nonzero(np.not_equal(mean_pred[start:end].argmax(axis=1),
                                                     data_y.argmax(axis=1)))
            accuracy = 1 - (err / mean_pred.shape[0])
            print('Angle: {0}, Accuracy: {1:.02%}'.format(angle, accuracy))
            for step in range(100):
                idx = np.array(range(step * 100, (step+1)*100))
                plot_(add_augment(self.data_reader.x_test[idx], mode='rotate', angle=angle),
                      self.data_reader.y_test[idx], accuracy, pred_mean=mean_pred[idx], pred_var=var_pred[idx],
                      mean_entropy=np.mean(var_pred), save_path=IMAGE_DIR + str(step), value=angle)


def predictive_entropy(prob):
    """
    Entropy of the probabilities (to measure the epistemic uncertainty)
    :param prob: probabilities of shape [batch_size, C]
    :return: Entropy of shape [batch_size]
    """
    eps = 1e-5
    return -1 * np.sum(np.log(prob+eps) * prob, axis=1)


def mutual_info(mean_prob, mc_prob):
    """
    computes the mutual information
    :param mean_prob: average MC probabilities of shape [batch_size, num_cls]
    :param mc_prob: List MC probabilities of length mc_simulations;
                    each of shape  of shape [batch_size, num_cls]
    :return: mutual information of shape [batch_size, num_cls]
    """
    eps = 1e-5
    first_term = -1 * np.sum(mean_prob * np.log(mean_prob + eps), axis=1)
    second_term = np.sum(np.mean([prob * np.log(prob + eps) for prob in mc_prob], axis=0), axis=1)
    return first_term + second_term


def compute_uncertainty_metrics(y_ohe, y_pred_ohe, y_var, desired_threshold):
    y = np.argmax(y_ohe, axis=1)
    y_pred = np.argmax(y_pred_ohe, axis=1)

    umin = np.min(y_var)
    umax = np.max(y_var)
    N_tot = np.prod(y.shape)
    wrong_pred = (y != y_pred).astype(int)
    right_pred = (y == y_pred).astype(int)

    ut = (umax - umin) * desired_threshold + umin

    uncertain = (y_var >= ut).astype(int)
    certain = (y_var < ut).astype(int)
    TP = np.sum(uncertain * wrong_pred)
    TN = np.sum(certain * right_pred)
    N_w = np.sum(wrong_pred)
    N_c = np.sum(certain)
    N_unc = np.sum(uncertain)
    recall = TP / N_w
    npv = TN / N_c
    acc = (TN + TP) / N_tot
    return recall, npv, acc

