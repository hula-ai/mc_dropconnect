import tensorflow as tf
from tqdm import tqdm

from DataLoaders.Data_Loader_3D import DataLoader
from utils.plot_utils import plot_save_preds_3d
from utils.loss_utils import cross_entropy, dice_coeff, weighted_cross_entropy
from utils.eval_utils import get_hist, compute_iou, var_calculate_3d, get_uncertainty_precision
import os
import numpy as np


class BaseModel(object):

    def __init__(self, sess, conf):
        self.sess = sess
        self.conf = conf
        self.bayes = conf.bayes
        self.input_shape = [None, None, None, None, self.conf.channel]
        self.output_shape = [None, None, None, None]
        self.create_placeholders()

    def create_placeholders(self):
        with tf.name_scope('Input'):
            self.inputs_pl = tf.placeholder(tf.float32, self.input_shape, name='input')
            self.labels_pl = tf.placeholder(tf.int64, self.output_shape, name='annotation')
            self.is_training_pl = tf.placeholder(tf.bool, name="is_training")
            self.with_dropout_pl = tf.placeholder(tf.bool, name="with_dropout")
            self.keep_prob_pl = tf.placeholder(tf.float32)

    def loss_func(self):
        with tf.name_scope('Loss'):
            self.y_prob = tf.nn.softmax(self.logits, axis=-1)
            y_one_hot = tf.one_hot(self.labels_pl, depth=self.conf.num_cls, axis=4, name='y_one_hot')
            if self.conf.weighted_loss:
                loss = weighted_cross_entropy(y_one_hot, self.logits, self.conf.num_cls, data=self.conf.data)
            else:
                if self.conf.loss_type == 'cross-entropy':
                    with tf.name_scope('cross_entropy'):
                        loss = cross_entropy(y_one_hot, self.logits, self.conf.num_cls)
                elif self.conf.loss_type == 'dice':
                    with tf.name_scope('dice_coefficient'):
                        loss = dice_coeff(y_one_hot, self.logits)
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
            self.y_pred = tf.argmax(self.logits, axis=4, name='decode_pred')
            correct_prediction = tf.equal(self.labels_pl, self.y_pred, name='correct_pred')
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
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
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
        if self.conf.random_crop:
            slice = int(self.conf.crop_size[-1] / 2)
        else:
            slice = int(self.conf.depth / 2)
        summary_list = [tf.summary.scalar('learning_rate', self.learning_rate),
                        tf.summary.scalar('loss', self.mean_loss),
                        tf.summary.scalar('accuracy', self.mean_accuracy),
                        tf.summary.image('train/original_image',
                                         self.inputs_pl[:, :, :, slice, :],
                                         max_outputs=5),
                        tf.summary.image('train/prediction_mask',
                                         tf.cast(tf.expand_dims(self.y_pred[:, :, :, slice], -1),
                                                 tf.float32),
                                         max_outputs=5),
                        tf.summary.image('train/original_mask',
                                         tf.cast(tf.expand_dims(self.labels_pl[:, :, :, slice], -1), tf.float32),
                                         max_outputs=5)]
        self.merged_summary = tf.summary.merge(summary_list)

    def save_summary(self, summary, step, is_train):
        # print('----> Summarizing at step {}'.format(step))
        if is_train:
            self.train_writer.add_summary(summary, step)
        else:
            self.valid_writer.add_summary(summary, step)
        self.sess.run(tf.local_variables_initializer())

    def train(self):
        self.sess.run(tf.local_variables_initializer())
        self.best_validation_loss = 1000
        if self.conf.reload_step > 0:
            self.reload(self.conf.reload_step)
            print('----> Continue Training from step #{}'.format(self.conf.reload_step))
        else:
            print('----> Start Training')
        self.data_reader = DataLoader(self.conf)
        self.numValid = self.data_reader.count_num_samples(mode='valid')
        self.num_val_batch = int(self.numValid / self.conf.val_batch_size)
        for train_step in range(self.conf.reload_step, self.conf.reload_step + self.conf.max_step + 1):
            x_batch, y_batch = self.data_reader.next_batch(mode='train')
            feed_dict = {self.inputs_pl: x_batch,
                         self.labels_pl: y_batch,
                         self.is_training_pl: True,
                         self.with_dropout_pl: True,
                         self.keep_prob_pl: self.conf.keep_prob}
            if train_step % self.conf.SUMMARY_FREQ == 0:
                _, _, _, summary = self.sess.run([self.train_op,
                                                  self.mean_loss_op,
                                                  self.mean_accuracy_op,
                                                  self.merged_summary],
                                                 feed_dict=feed_dict)
                loss, acc = self.sess.run([self.mean_loss, self.mean_accuracy])
                print('step: {0:<6}, train_loss= {1:.4f}, train_acc={2:.01%}'.format(train_step, loss, acc))
                self.save_summary(summary, train_step, is_train=True)
            else:
                self.sess.run([self.train_op, self.mean_loss_op, self.mean_accuracy_op], feed_dict=feed_dict)
            if train_step % self.conf.VAL_FREQ == 0:
                print('-' * 25 + 'Validation' + '-' * 25)
                self.normal_evaluate(dataset='valid', train_step=train_step)

    def test(self, step_num):
        self.sess.run(tf.local_variables_initializer())

        print('loading the model.......')
        self.reload(step_num)

        self.data_reader = DataLoader(self.conf)
        self.numTest = self.data_reader.count_num_samples(mode='test')
        self.num_test_batch = int(self.numTest / self.conf.val_batch_size)

        print('-' * 25 + 'Test' + '-' * 25)
        if not self.conf.bayes:
            self.normal_evaluate(dataset='test', train_step=step_num)
        else:
            self.MC_evaluate(dataset='test', train_step=step_num)
        # self.visualize(num_samples=20, train_step=step_num, mode='test')

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

    def normal_evaluate(self, dataset='valid', train_step=None):
        num_batch = self.num_test_batch if dataset == 'test' else self.num_val_batch
        self.sess.run(tf.local_variables_initializer())
        hist = np.zeros((self.conf.num_cls, self.conf.num_cls))
        scan_num = 0
        for image_index in range(num_batch):
            data_x, data_y = self.data_reader.next_batch(num=scan_num, mode=dataset)
            depth = data_x.shape[0] * data_x.shape[-2]
            scan_input = np.zeros((self.conf.height, self.conf.width, depth, self.conf.channel))
            scan_mask = np.zeros((self.conf.height, self.conf.width, depth))
            scan_mask_pred = np.zeros((self.conf.height, self.conf.width, depth))
            for slice_num in range(data_x.shape[0]):  # for each slice of the 3D image
                feed_dict = {self.inputs_pl: np.expand_dims(data_x[slice_num], 0),
                             self.labels_pl: np.expand_dims(data_y[slice_num], 0),
                             self.is_training_pl: True,
                             self.with_dropout_pl: False,
                             self.keep_prob_pl: 1}
                self.sess.run([self.mean_loss_op, self.mean_accuracy_op], feed_dict=feed_dict)
                inputs, mask, mask_pred = self.sess.run([self.inputs_pl,
                                                         self.labels_pl,
                                                         self.y_pred], feed_dict=feed_dict)
                hist += get_hist(mask_pred.flatten(), mask.flatten(), num_cls=self.conf.num_cls)
                idx_d, idx_u = slice_num * self.conf.Dcut_size, (slice_num + 1) * self.conf.Dcut_size
                scan_input[:, :, idx_d:idx_u] = np.squeeze(inputs, axis=0)
                scan_mask[:, :, idx_d:idx_u] = np.squeeze(mask, axis=0)
                scan_mask_pred[:, :, idx_d:idx_u] = np.squeeze(mask_pred, axis=0)
            self.visualize_me(np.squeeze(scan_input), scan_mask, scan_mask_pred, train_step=train_step,
                              img_idx=image_index, mode='valid')
            scan_num += 1
        IOU, ACC = compute_iou(hist)
        mean_IOU = np.mean(IOU)
        loss, acc = self.sess.run([self.mean_loss, self.mean_accuracy])

        if dataset == "valid":  # save the summaries and improved model in validation mode
            summary_valid = self.sess.run(self.merged_summary, feed_dict=feed_dict)
            self.save_summary(summary_valid, train_step, is_train=False)
            if loss < self.best_validation_loss:
                self.best_validation_loss = loss
                print('>>>>>>>> model validation loss improved; saving the model......')
                self.save(train_step)

        print('After {0} training step: val_loss= {1:.4f}, val_acc={2:.01%}'.format(train_step, loss, acc))
        print('- IOU: bg={0:.01%}, liver={1:.01%}, spleen={2:.01%}, '
              'kidney={3:.01%}, bone={4:.01%}, vessel={5:.01%}, mean_IoU={6:.01%}'
              .format(IOU[0], IOU[1], IOU[2], IOU[3], IOU[4], IOU[5], mean_IOU))
        print('- ACC: bg={0:.01%}, liver={1:.01%}, spleen={2:.01%}, '
              'kidney={3:.01%}, bone={4:.01%}, vessel={5:.01%}'
              .format(ACC[0], ACC[1], ACC[2], ACC[3], ACC[4], ACC[5]))
        print('-' * 60)

    def MC_evaluate(self, dataset='valid', train_step=None):
        # num_batch = self.num_test_batch if dataset == 'test' else self.num_val_batch
        # hist = np.zeros((self.conf.num_cls, self.conf.num_cls))
        # self.sess.run(tf.local_variables_initializer())
        # scan_num = 0
        # for image_index in tqdm(range(num_batch)[:2]):
        #     data_x, data_y = self.data_reader.next_batch(num=scan_num, mode=dataset)
        #     depth = data_x.shape[0] * data_x.shape[-2]
        #     scan_input = np.zeros((self.conf.height, self.conf.width, depth, self.conf.channel))
        #     scan_mask = np.zeros((self.conf.height, self.conf.width, depth))
        #     scan_mask_prob = np.zeros((self.conf.height, self.conf.width, depth, self.conf.num_cls))
        #     scan_mask_pred = np.zeros((self.conf.height, self.conf.width, depth))
        #     scan_mask_pred_mc = [np.zeros_like(scan_mask_pred) for _ in range(self.conf.monte_carlo_simulations)]
        #     scan_mask_prob_mc = [np.zeros_like(scan_mask_prob) for _ in range(self.conf.monte_carlo_simulations)]
        #     for slice_num in range(data_x.shape[0]):  # for each slice of the 3D image
        #         idx_d, idx_u = slice_num * self.conf.Dcut_size, (slice_num + 1) * self.conf.Dcut_size
        #         for mc_iter in range(self.conf.monte_carlo_simulations):
        #             feed_dict = {self.inputs_pl: np.expand_dims(data_x[slice_num], 0),
        #                          self.labels_pl: np.expand_dims(data_y[slice_num], 0),
        #                          self.is_training_pl: True,
        #                          self.with_dropout_pl: True,
        #                          self.keep_prob_pl: self.conf.keep_prob}
        #             inputs, mask, mask_prob, mask_pred = self.sess.run([self.inputs_pl,
        #                                                                 self.labels_pl,
        #                                                                 self.y_prob,
        #                                                                 self.y_pred], feed_dict=feed_dict)
        #             scan_mask_prob_mc[mc_iter][:, :, idx_d:idx_u] = np.squeeze(mask_prob, axis=0)
        #             scan_mask_pred_mc[mc_iter][:, :, idx_d:idx_u] = np.squeeze(mask_pred, axis=0)
        #         scan_input[:, :, idx_d:idx_u] = np.squeeze(inputs, axis=0)
        #         scan_mask[:, :, idx_d:idx_u] = np.squeeze(mask, axis=0)
        #
        #     prob_mean = np.nanmean(scan_mask_prob_mc, axis=0)
        #     prob_variance = np.var(scan_mask_prob_mc, axis=0)
        #     pred = np.argmax(prob_mean, axis=-1)
        #     var_one = var_calculate_3d(pred, prob_variance)
        #     hist += get_hist(pred.flatten(), scan_mask.flatten(), num_cls=self.conf.num_cls)
        #     self.visualize_me(np.squeeze(scan_input), scan_mask, pred, var_one, train_step=train_step,
        #                       img_idx=image_index, mode='test')
        #     scan_num += 1
        #
        import h5py
        h5f = h5py.File(self.conf.run_name + '_bayes.h5', 'r')
        # h5f.create_dataset('x', data=all_inputs)
        # h5f.create_dataset('y', data=all_mask)
        # h5f.create_dataset('y_pred', data=all_pred)
        # h5f.create_dataset('y_var', data=all_var)
        # h5f.create_dataset('cls_uncertainty', data=cls_uncertainty)
        # h5f.close()

        all_mask = h5f['y'][:]
        all_pred = h5f['y_pred'][:]
        all_var = h5f['y_var'][:]
        h5f.close()
        uncertainty_measure = get_uncertainty_precision(all_mask, all_pred, all_var)
        print('Uncertainty Quality Measure = {}'.format(uncertainty_measure))
        IOU, ACC = compute_iou(hist)
        mean_IOU = np.mean(IOU)
        print('****** IoU & ACC ******')
        print('Mean IoU = {0:.01%}'.format(mean_IOU))
        for ii in range(self.conf.num_cls):
            print('     - {0} class: IoU={1:.01%}, ACC={2:.01%}'.format(self.conf.label_name[ii], IOU[ii], ACC[ii]))
        print('-' * 20)

    def visualize_me(self, x, y, y_pred, var=None, train_step=None, img_idx=None,
                     mode='valid'):  # all of shape (512, 512, num_slices)
        depth = y.shape[-1]
        slices = np.linspace(20, depth - 20, 10).astype(int)
        x_plot = [x[:, :, i] for i in slices]
        y_plot = [y[:, :, i] for i in slices]
        pred_plot = [y_pred[:, :, i] for i in slices]
        if mode == 'valid':
            dest_path = os.path.join(self.conf.imagedir + self.conf.run_name, str(train_step), str(img_idx))
        elif mode == "test":
            dest_path = os.path.join(self.conf.imagedir + self.conf.run_name, str(train_step) + '_test', str(img_idx))

        print('saving sample prediction images....... ')

        if not self.conf.bayes or mode == 'valid':
            # run it either in validation mode or when non-bayesian network
            plot_save_preds_3d(x_plot, y_plot, pred_plot, slice_numbers=slices, depth=depth,
                               path=dest_path, label_names=np.array(self.conf.label_name))
        else:
            var_plot = [var[:, :, i] for i in slices]
            plot_save_preds_3d(x_plot, y_plot, pred_plot, var_plot, slice_numbers=slices, depth=depth,
                               path=dest_path + '/bayes', label_names=np.array(self.conf.label_name))

    def visualize(self, num_samples, train_step, mode='valid'):

        scan_index = np.random.randint(low=0, high=len(self.pred_), size=num_samples)
        slice_index = np.array([np.random.randint(low=0, high=self.pred_[si].shape[-1])
                                for si in scan_index])
        if mode == 'valid':
            dest_path = os.path.join(self.conf.imagedir + self.conf.run_name, str(train_step))
        elif mode == "test":
            dest_path = os.path.join(self.conf.imagedir + self.conf.run_name, str(train_step) + '_test')

        x_plot = np.concatenate([np.expand_dims(self.input_[scan_idx][:, :, slice_idx].squeeze(), axis=0)
                                 for scan_idx, slice_idx in zip(scan_index, slice_index)], axis=0)
        y_plot = np.concatenate([np.expand_dims(self.label_[scan_idx][:, :, slice_idx].squeeze(), axis=0)
                                 for scan_idx, slice_idx in zip(scan_index, slice_index)], axis=0)
        pred_plot = np.concatenate([np.expand_dims(self.pred_[scan_idx][:, :, slice_idx].squeeze(), axis=0)
                                    for scan_idx, slice_idx in zip(scan_index, slice_index)], axis=0)
        print('saving sample prediction images....... ')

        if not self.conf.bayes or mode == 'valid':
            # run it either in validation mode or when non-bayesian network
            plot_save_preds(x_plot, y_plot, pred_plot, path=dest_path, label_names=np.array(self.conf.label_name))
        else:
            var_plot = np.concatenate([np.expand_dims(self.pred_var[scan_idx][:, :, slice_idx].squeeze(), axis=0)
                                       for scan_idx, slice_idx in zip(scan_index, slice_index)], axis=0)
            plot_save_preds(x_plot, y_plot, pred_plot, var_plot, dest_path + 'bayes', np.array(self.conf.label_name))
