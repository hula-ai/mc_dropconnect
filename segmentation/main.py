import tensorflow as tf
from config import args
import os
from utils.other_utils import write_spec


if args.dim == 2:
    from model.model_2D.SegNet_DropConnect import SegNet as Model
else:
    from model.model_3D.VNet import VNet as Model


def main(_):
    if args.mode not in ['train', 'test', 'predict']:
        print('invalid mode: ', args.mode)
        print("Please input a mode: train, test, or predict")
    else:
        model = Model(tf.Session(), args)
        if not os.path.exists(args.modeldir+args.run_name):
            os.makedirs(args.modeldir+args.run_name)
        if not os.path.exists(args.logdir+args.run_name):
            os.makedirs(args.logdir+args.run_name)
        if args.mode == 'train':
            write_spec(args)
            model.train()
        elif args.mode == 'test':
            model.test(step_num=args.reload_step)


if __name__ == '__main__':
    # configure which gpu or cpu to use
    os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2'
    tf.app.run()
