"""
Settings.
"""


import argparse


pretrained_weights_frame = "./models/pretrained_models/frame_best.pth"
pretrained_weights_sound = "./models/pretrained_models/sound_best.pth"


class ArgParserTest(object):
    def __init__(self):
        parser = argparse.ArgumentParser()
        # Misc arguments
        parser.add_argument('--mode', default='eval',
                            help="train/eval")
        parser.add_argument('--seed', default=1234, type=int,
                            help='manual seed')
        parser.add_argument('--ckpt', default='./test',
                            help='folder to output checkpoints')
        parser.add_argument('--disp_iter', type=int, default=400,
                            help='frequency to display')
        parser.add_argument('--eval_epoch', type=int, default=1,
                            help='frequency to evaluate')
        parser.add_argument('--log', default=None,
                            help='the file to store the training log')

        # Model related arguments
        parser.add_argument('--id', default='',
                            help="a name for identifying the model")
        parser.add_argument('--num_mix', default=2, type=int,
                            help="number of sounds to mix")
        parser.add_argument('--arch_sound', default='pcnetlr',
                            help="architecture of net_sound")
        parser.add_argument('--arch_frame', default='resnet18fc',
                            help="architecture of net_frame")
        parser.add_argument('--weights_frame', default=pretrained_weights_frame,
                            help="weights to finetune net_frame")
        parser.add_argument('--weights_sound', default=pretrained_weights_sound,
                            help="weights to finetune net_sound")
        parser.add_argument('--num_frames', default=3, type=int,
                            help='number of frames')
        parser.add_argument('--stride_frames', default=24, type=int,
                            help='sampling stride of frames')
        parser.add_argument('--output_activation', default='sigmoid',
                            help="activation on the output")
        parser.add_argument('--binary_mask', default=1, type=int,
                            help="whether to use bianry masks")
        parser.add_argument('--mask_thres', default=0.5, type=float,
                            help="threshold in the case of binary masks")
        parser.add_argument('--loss', default='bce',
                            help="loss function to reconstruct target mask")
        parser.add_argument('--weighted_loss', default=1, type=int,
                            help="weighted loss")
        parser.add_argument('--log_freq', default=1, type=int,
                            help="log frequency scale")

        # SimIter related arguments
        parser.add_argument('--cycles_inner', default=4, type=int,
                            help='number of inner cycles to update representations in PC')
        parser.add_argument('--cycs_in_test', default=4, type=int,
                            help='number of inner cycles to update representations in PC at test stage')
        parser.add_argument('--n_fm_visual', default=16, type=int,
                            help='number of visual feature maps predicted in PC')
        parser.add_argument('--n_fm_out', default=1, type=int,
                            help='number of output feature maps in PC')

        # Distributed Data Parallel
        parser.add_argument('--gpu_ids', default='0,1', type=str)
        parser.add_argument('--num_gpus', default=2, type=int,
                            help='number of gpus to use within a node')
        parser.add_argument('--batch_size_per_gpu', default=32, type=int,
                            help='input batch size')
        parser.add_argument('--workers', default=8, type=int,
                            help='number of data loading workers')
        parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N',
                            help='number of nodes for distributed training')
        parser.add_argument('-nr', '--nr', default=0, type=int,
                            help='node rank for distributed training')

        # Data related arguments
        parser.add_argument('--audLen', default=65535, type=int,
                            help='sound length')
        parser.add_argument('--audRate', default=11025, type=int,
                            help='sound sampling rate')
        parser.add_argument('--stft_frame', default=1022, type=int,
                            help="stft frame length")
        parser.add_argument('--stft_hop', default=256, type=int,
                            help="stft hop length")
        parser.add_argument('--imgSize', default=224, type=int,
                            help='size of input frame')
        parser.add_argument('--frameRate', default=8, type=float,
                            help='video frame sampling rate')
        parser.add_argument('--num_val', default=-1, type=int,
                            help='number of images to evalutate')
        parser.add_argument('--num_test', default=-1, type=int,
                            help='number of images to test')
        parser.add_argument('--list_train', default='./data/train516.csv')
        parser.add_argument('--list_val', default='./data/val11.csv')
        parser.add_argument('--list_test', default='./data/test11.csv')
        parser.add_argument('--dup_trainset', default=100, type=int,
                            help='duplicate so that one epoch has more iters')
        parser.add_argument('--dup_validset', default=10, type=int,
                            help='duplicate so that validation results would be more meaningful')
        parser.add_argument('--dup_testset', default=10, type=int,
                            help='duplicate so that test results would be more meaningful')

        # Optimization related arguments
        parser.add_argument('--num_epoch', default=100, type=int,
                            help='epochs to train for')
        parser.add_argument('--lr_frame', default=1e-4, type=float, help='LR')
        parser.add_argument('--lr_sound', default=1e-3, type=float, help='LR')
        parser.add_argument('--lr_steps', nargs='+', type=int, default=[40, 80],
                            help='steps to drop LR in epochs')
        parser.add_argument('--beta1', default=0.9, type=float,
                            help='momentum for sgd, beta1 for adam')
        parser.add_argument('--weight_decay', default=1e-2, type=float,
                            help='weights regularizer')

        self.parser = parser

    def parse_test_arguments(self):
        args = self.parser.parse_args()

        print("Input arguments:")
        for key, val in vars(args).items():
            print("{:16} {}".format(key, val))

        return args
