import os
from config import args
import tensorflow as tf
if args.model == 'lenet':
    if args.run_name == 'none':
        from models.lenet import LeNet as Model
    elif args.run_name == 'dropout':
        from models.lenet_dropout import LeNet as Model
    elif args.run_name == 'dropconnect':
        from models.lenet_dropconnect import LeNet as Model
elif args.model == 'fcnet':
    if args.run_name == "dropout":
        from models.FCNet_dropout import FCNet as Model
    elif args.run_name == "dropconnect":
        from models.FCNet_dropconnect import FCNet as Model
    elif args.run_name == "none":
        from models.FCNet import FCNet as Model


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    model = Model(tf.Session(), args)
    if not os.path.exists(args.modeldir + args.run_name):
        os.makedirs(args.modeldir + args.run_name)
    if not os.path.exists(args.logdir + args.run_name):
        os.makedirs(args.logdir + args.run_name)
    if args.mode == 'train':
        model.train()
    elif args.mode == 'test':
        model.test(step_num=args.reload_step)
