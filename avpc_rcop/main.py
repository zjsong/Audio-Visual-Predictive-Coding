import os
import time
import random
import warnings

import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp

from arguments import ArgParser
from dataset import MUSICMixDataset
from models import ModelBuilder, activate
from utils import AverageMeter, warpgrid, makedirs, output_visuals, calc_metrics, plot_loss_metrics

warnings.filterwarnings("ignore")


def main():
    # arguments
    parser = ArgParser()
    args = parser.parse_train_arguments()
    args.batch_size = args.num_gpus * args.batch_size_per_gpu

    if torch.cuda.is_available():
        args.device = torch.device("cuda")
    else:
        args.device = torch.device("cpu")

    # experiment name
    if args.mode == 'train':
        args.id += '-{}mix'.format(args.num_mix_train_pt)
        if args.log_freq:
            args.id += '-LogFreq'
        args.id += '-{}-{}'.format(args.arch_frame, args.arch_sound)
        args.id += '-frames{}stride{}'.format(args.num_frames, args.stride_frames)
        if args.binary_mask:
            assert args.loss == 'bce', 'Binary Mask should go with BCE loss'
            args.id += '-binary'
        else:
            args.id += '-ratio'
        if args.weighted_loss:
            args.id += '-weightedLoss'

    print('Model ID: {}'.format(args.id))

    # paths to save/load output
    args.vis_pt = os.path.join(args.ckpt, 'pre-train/visualization/')
    args.log_pt = os.path.join(args.ckpt, 'running_log_pt.txt')
    args.vis_ft = os.path.join(args.ckpt, 'fine-tune/visualization/')
    args.log_ft = os.path.join(args.ckpt, 'running_log_ft.txt')

    # initialize the best loss_simsiam with a big number
    args.best_loss_simsiam = float("inf")
    # initialize the best SDR with a small number
    args.best_sdr = -float("inf")

    args.world_size = args.num_gpus * args.nodes
    os.environ['MASTER_ADDR'] = 'xxx.xx.xx.xx'  # specified by yourself
    os.environ['MASTER_PORT'] = 'xxxx'  # specified by yourself
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    mp.spawn(main_worker, nprocs=args.num_gpus, args=(args,))


def main_worker(gpu, args):
    rank = args.nr * args.num_gpus + gpu
    dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=rank)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # network builders
    builder = ModelBuilder()
    net_frame = builder.build_frame(
        arch=args.arch_frame,
        fc_vis=args.n_fm_visual,
        weights=args.weights_frame)
    net_sound = builder.build_sound(
        arch=args.arch_sound,
        weights=args.weights_sound,
        cyc_in=args.cycles_inner,
        fc_vis=args.n_fm_visual,
        n_fm_out=args.n_fm_out)
    net_simsiam = builder.build_simsiam(
        in_dim_proj=args.in_dim_projector,
        weights=args.weights_simsiam)

    if gpu == 0:
        # count number of parameters
        n_params_net_frame = sum(p.numel() for p in net_frame.parameters())
        print('#P of net_frame: {}'.format(n_params_net_frame))
        n_params_net_sound = sum(p.numel() for p in net_sound.parameters())
        print('#P of net_sound: {}'.format(n_params_net_sound))
        n_params_net_simsiam = sum(p.numel() for p in net_simsiam.parameters())
        print('#P of net_simsiam: {}'.format(n_params_net_simsiam))
        print('Total #P: {}'.format(n_params_net_frame + n_params_net_sound))   # net_simsiam is not used during inference

    # loss function
    crit_mask = builder.build_criterion(arch=args.loss)

    torch.cuda.set_device(gpu)
    net_frame.cuda(gpu)
    net_sound.cuda(gpu)
    net_simsiam.cuda(gpu)

    # wrap model
    netWrapper = NetWrapper(net_frame, net_sound, net_simsiam, crit_mask)
    netWrapper = torch.nn.parallel.DistributedDataParallel(netWrapper, device_ids=[gpu], find_unused_parameters=True)
    netWrapper.to(args.device)

    # set up optimizer
    optimizer_pt = create_optimizer(net_frame, net_sound, net_simsiam, args, include_pretrain=True)
    optimizer_ft = create_optimizer(net_frame, net_sound, net_simsiam, args, include_pretrain=False)

    args.batch_size_ = int(args.batch_size / args.num_gpus)
    args.batch_size_val_test = 30

    # dataset and loader
    dataset_train_pt_solo = MUSICMixDataset(args.list_train, args, process_stage='train_pt')
    dataset_train_ft_solo = MUSICMixDataset(args.list_train, args, process_stage='train_ft')
    dataset_val = MUSICMixDataset(args.list_val, args, max_sample=args.num_val, process_stage='val')
    dataset_test = MUSICMixDataset(args.list_test, args, max_sample=args.num_test, process_stage='test')

    train_sampler_pt_solo = torch.utils.data.distributed.DistributedSampler(dataset_train_pt_solo,
                                                                            num_replicas=args.world_size,
                                                                            rank=rank)
    train_sampler_ft_solo = torch.utils.data.distributed.DistributedSampler(dataset_train_ft_solo,
                                                                            num_replicas=args.world_size,
                                                                            rank=rank)

    loader_train_pt_solo = torch.utils.data.DataLoader(
        dataset_train_pt_solo,
        batch_size=args.batch_size_,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler_pt_solo,
        drop_last=True)
    loader_train_ft_solo = torch.utils.data.DataLoader(
        dataset_train_ft_solo,
        batch_size=args.batch_size_,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler_ft_solo,
        drop_last=True)
    loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size_val_test,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False)
    loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size_val_test,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False)

    # ------------------------------------------------
    # stage 1: pre-training with SimSiam architecture
    # ------------------------------------------------
    # history of performance
    history_pt = {
        'train': {'epoch': [], 'loss': []},
        'val': {'epoch': [], 'loss': []}}

    if gpu == 0:
        args.epoch_iters = len(dataset_train_pt_solo) // args.batch_size
        print('1 Epoch = {} iters'.format(args.epoch_iters))

        # eval first
        evaluate_pt(netWrapper, loader_val, history_pt, 0, args)
        evaluate_testset(netWrapper, loader_test, 0, args)
        print('Evaluation before training is done!')

    # start pre-training
    for epoch in range(args.num_epoch_pt):

        # pre-training with only solo data
        train_pt(netWrapper, loader_train_pt_solo, optimizer_pt, history_pt, epoch + 1, gpu, args)

        # evaluation and visualization
        if epoch % args.eval_epoch == 0 and gpu == 0:
            evaluate_pt(netWrapper, loader_val, history_pt, epoch + 1, args)

            # checkpointing
            checkpoint(net_frame, net_sound, net_simsiam, args, include_pretrain=True)
            print('Saving checkpoints at epoch {}.'.format(epoch + 1))

        # adjust learning rate
        if epoch + 1 in args.lr_steps_pt:
            adjust_learning_rate(optimizer_pt, args, include_pretrain=True)

    if gpu == 0:
        print('\nPre-training Done!')

    # ------------------------------------------------
    # stage 2: fine-tuning with GT masks
    # ------------------------------------------------
    # history of performance
    history_ft = {
        'train': {'epoch': [], 'loss': []},
        'val': {'epoch': [], 'loss': [], 'sdr': [], 'sir': [], 'sar': []}}

    if gpu == 0:
        # eval first
        evaluate_ft(netWrapper, loader_val, history_ft, 0, args)
        evaluate_testset(netWrapper, loader_test, 0, args)
        print('Evaluation before training is done!')

    # start training
    for epoch in range(args.num_epoch_ft):

        train_ft(netWrapper, loader_train_ft_solo, optimizer_ft, history_ft, epoch + 1, gpu, args)

        # evaluation and visualization
        if epoch % args.eval_epoch == 0 and gpu == 0:
            evaluate_ft(netWrapper, loader_val, history_ft, epoch + 1, args)

            # save checkpoint and test modal on test set
            cur_sdr = history_ft['val']['sdr'][-1]
            if cur_sdr > args.best_sdr:
                args.best_sdr = cur_sdr

                checkpoint(net_frame, net_sound, net_simsiam, args)
                print('Saving checkpoints with the best validation performance at epoch {}.'.format(epoch + 1))

                evaluate_testset(netWrapper, loader_test, epoch + 1, args)

        # adjust learning rate
        if epoch + 1 in args.lr_steps_ft:
            adjust_learning_rate(optimizer_ft, args)

    if gpu == 0:
        print('\nFine-tuning Done!')


# network wrapper, defines forward pass
class NetWrapper(torch.nn.Module):
    def __init__(self, net_frame, net_sound, net_simsiam, crit):
        super(NetWrapper, self).__init__()
        self.net_frame, self.net_sound, self.net_simsiam = net_frame, net_sound, net_simsiam
        self.crit = crit

    def forward(self, batch_data, args, include_pretrain=False, train_eval='train'):

        if include_pretrain:

            if train_eval == 'train':
                mag_mix = batch_data['mag_mix']
                frames = batch_data['frames'][0].cuda(non_blocking=True)

                N = args.num_mix_train_pt
                B = mag_mix[0].size(0)
                T = mag_mix[0].size(3)

                # 0.0 warp the spectrogram
                log_mag_mix = [None for n in range(N - 1)]
                if args.log_freq:
                    grid_warp = torch.from_numpy(warpgrid(B, 256, T, warp=True)).cuda(non_blocking=True)
                    for n in range(N-1):
                        mag_mix[n] = mag_mix[n] + 1e-10
                        mag_mix[n] = mag_mix[n].cuda(non_blocking=True)
                        mag_mix[n] = F.grid_sample(mag_mix[n], grid_warp, align_corners=False)
                        log_mag_mix[n] = torch.log(mag_mix[n]).detach()
                else:
                    for n in range(N-1):
                        mag_mix[n] = mag_mix[n] + 1e-10
                        mag_mix[n] = mag_mix[n].cuda(non_blocking=True)
                        log_mag_mix[n] = torch.log(mag_mix[n]).detach()

                # 1. forward net_frame
                # only compute feature maps of the first video clip
                feat_map_frames = self.net_frame.forward_multiframe(frames)

                # 2. forward net_sound -> BxCxHxW
                feat_sound_from_mix = [None for n in range(N-1)]
                for n in range(N-1):
                    _, feat_sound_from_mix[n] = self.net_sound(log_mag_mix[n], feat_map_frames)

                # 3. prediction in SimSiam
                distance_pred = 0
                for n in range(N-2):
                    distance_pred = distance_pred + self.net_simsiam.forward(feat_sound_from_mix[n],
                                                                             feat_sound_from_mix[n+1])

                return distance_pred

            else:
                mag_mix = batch_data['mag_mix']
                mags = batch_data['mags'][0]
                frames = batch_data['frames'][0].cuda(non_blocking=True)
                mag_mix = (mag_mix + 1e-10).cuda(non_blocking=True)
                mags = (mags + 1e-10).cuda(non_blocking=True)

                B = mag_mix.size(0)
                T = mag_mix.size(3)

                # 0.0 warp the spectrogram
                if args.log_freq:
                    grid_warp = torch.from_numpy(warpgrid(B, 256, T, warp=True)).cuda(non_blocking=True)
                    mag_mix = F.grid_sample(mag_mix, grid_warp, align_corners=False)
                    mags = F.grid_sample(mags, grid_warp, align_corners=False)

                log_mag_mix = torch.log(mag_mix).detach()
                log_mags = torch.log(mags).detach()

                # 1. forward net_frame
                # only compute feature maps of the first video clip
                feat_map_frames = self.net_frame.forward_multiframe(frames)

                # 2. forward net_sound -> BxCxHxW
                _, feat_sound_from_mix = self.net_sound(log_mag_mix, feat_map_frames)
                _, feat_sound_from_single = self.net_sound(log_mags, feat_map_frames)

                # 3. prediction in SimSiam
                distance_pred = self.net_simsiam.forward(feat_sound_from_mix, feat_sound_from_single)

                return distance_pred

        else:
            mag_mix = batch_data['mag_mix']
            mags = batch_data['mags']
            frames = batch_data['frames']
            mag_mix = mag_mix + 1e-10

            N = args.num_mix_train_ft
            B = mag_mix.size(0)
            T = mag_mix.size(3)

            mag_mix = mag_mix.cuda(non_blocking=True)
            for n in range(N):
                mags[n] = mags[n].cuda(non_blocking=True)
                frames[n] = frames[n].cuda(non_blocking=True)

            # 0.0 warp the spectrogram
            if args.log_freq:
                grid_warp = torch.from_numpy(warpgrid(B, 256, T, warp=True)).cuda(non_blocking=True)
                mag_mix = F.grid_sample(mag_mix, grid_warp, align_corners=False)
                for n in range(N):
                    mags[n] = F.grid_sample(mags[n], grid_warp, align_corners=False)

            # 0.1 calculate loss weighting coefficient: magnitude of input mixture
            if args.weighted_loss:
                weight = torch.log1p(mag_mix)
                weight = torch.clamp(weight, 1e-3, 10)
            else:
                weight = torch.ones_like(mag_mix)

            # 0.2 ground truth masks are computed after warpping!
            gt_masks = [None for n in range(N)]
            for n in range(N):
                if args.binary_mask:
                    # for simplicity, mag_N > 0.5 * mag_mix
                    gt_masks[n] = (mags[n] > 0.5 * mag_mix).float()
                else:
                    gt_masks[n] = mags[n] / mag_mix
                    # clamp to avoid large numbers in ratio masks
                    gt_masks[n].clamp_(0., 5.)

            # LOG magnitude
            log_mag_mix = torch.log(mag_mix).detach()

            # 1. forward net_frame
            feat_map_frames = [None for n in range(N)]
            for n in range(N):
                feat_map_frames[n] = self.net_frame.forward_multiframe(frames[n])

            # 2. forward net_sound
            pred_masks = [None for n in range(N)]
            for n in range(N):
                pred_masks[n], _ = self.net_sound(log_mag_mix, feat_map_frames[n])
                pred_masks[n] = activate(pred_masks[n], args.output_activation)

            # 3. loss
            err_mask = self.crit(pred_masks, gt_masks, weight).reshape(1)

            outputs = {'pred_masks': pred_masks, 'gt_masks': gt_masks,
                       'mag_mix': mag_mix, 'mags': mags, 'weight': weight}

            return err_mask, outputs


def train_pt(netWrapper, loader, optimizer_pt, history_pt, epoch, gpu, args):
    torch.set_grad_enabled(True)
    batch_time = AverageMeter()
    data_time = AverageMeter()

    # switch to train mode
    netWrapper.train()

    # main loop
    torch.cuda.synchronize()
    tic = time.perf_counter()
    for i, batch_data in enumerate(loader):
        # measure data time
        torch.cuda.synchronize()
        data_time.update(time.perf_counter() - tic)

        # forward pass
        netWrapper.zero_grad()

        loss_simsiam = netWrapper.forward(batch_data, args, include_pretrain=True, train_eval='train')
        loss = loss_simsiam.mean()

        # backward
        loss.backward()
        optimizer_pt.step()

        # measure total time
        torch.cuda.synchronize()
        batch_time.update(time.perf_counter() - tic)
        tic = time.perf_counter()

        # display
        if i % args.disp_iter == 0 and gpu == 0:
            if i == 0:
                print('\n------------------------------------------------------------------------------')
            print('Epoch: [{}][{}/{}], Time: {:.2f}, Data: {:.2f}, '
                  'lr_frame_pt: {}, lr_sound_pt: {}, lr_simsiam: {}, '
                  'loss_simsiam: {:.4f}'
                  .format(epoch, i, args.epoch_iters,
                          batch_time.average(), data_time.average(),
                          args.lr_frame_pt, args.lr_sound_pt, args.lr_simsiam,
                          loss.item()))

        if gpu == 0:
            fractional_epoch = epoch - 1 + 1. * i / args.epoch_iters
            history_pt['train']['epoch'].append(fractional_epoch)
            history_pt['train']['loss'].append(loss_simsiam.item())


def evaluate_pt(netWrapper, loader, history_pt, epoch, args):
    if epoch == 0:
        print('\n-------------------------------------- pre-training ----------------------------------------------------')
    print('Evaluating at epoch {}...'.format(epoch))
    torch.set_grad_enabled(False)

    # remove previous viz_pt results
    makedirs(args.vis_pt, remove=True)

    # switch to eval mode
    netWrapper.eval()

    # initialize meters
    loss_meter = AverageMeter()

    with torch.no_grad():
        for i, batch_data in enumerate(loader):
            loss_simsiam = netWrapper.forward(batch_data, args, include_pretrain=True, train_eval='eval')
            loss = loss_simsiam.mean()

            loss_meter.update(loss.item())

            total_batch = (11 * args.dup_validset // args.batch_size_val_test)
            if i == 0 or (i + 1) == (total_batch // 2) or (i + 1) == total_batch:
                print('[Eval] iter {}, loss_simsiam: {:.4f}'.format(i, loss.item()))

    metric_output = '[Eval Summary] Epoch: {}, Loss_SimSiam: {:.4f}'.format(epoch, loss_meter.average())
    print(metric_output)

    learning_rate = ' lr_sound: {}, lr_frame: {}, lr_simsiam: {}'.\
        format(args.lr_sound_pt, args.lr_frame_pt, args.lr_simsiam)
    with open(args.log_pt, 'a') as F:
        F.write(metric_output + learning_rate + '\n')

    history_pt['val']['epoch'].append(epoch)
    history_pt['val']['loss'].append(loss_meter.average())

    # Plot figure
    if epoch > 0:
        print('Plotting figures...')
        plot_loss_metrics(args.vis_pt, history_pt, include_pretrain=True)


# train one epoch
def train_ft(netWrapper, loader, optimizer_ft, history_ft, epoch, gpu, args):
    torch.set_grad_enabled(True)
    batch_time = AverageMeter()
    data_time = AverageMeter()

    # switch to train mode
    netWrapper.train()

    # main loop
    torch.cuda.synchronize()
    tic = time.perf_counter()
    for i, batch_data in enumerate(loader):
        # measure data time
        torch.cuda.synchronize()
        data_time.update(time.perf_counter() - tic)

        # forward pass
        netWrapper.zero_grad()

        err_mask, _ = netWrapper.forward(batch_data, args)
        loss = err_mask.mean()

        # backward
        loss.backward()
        optimizer_ft.step()

        # measure total time
        torch.cuda.synchronize()
        batch_time.update(time.perf_counter() - tic)
        tic = time.perf_counter()

        # display
        if i % args.disp_iter == 0 and gpu == 0:
            if i == 0:
                print('\n------------------------------------------------------------------------------')
            print('Epoch: [{}][{}/{}], Time: {:.2f}, Data: {:.2f}, '
                  'lr_frame: {}, lr_sound: {}, '
                  'loss_mask: {:.4f}'
                  .format(epoch, i, args.epoch_iters,
                          batch_time.average(), data_time.average(),
                          args.lr_frame_ft, args.lr_sound_ft,
                          loss.item()))

        if gpu == 0:
            fractional_epoch = epoch - 1 + 1. * i / args.epoch_iters
            history_ft['train']['epoch'].append(fractional_epoch)
            history_ft['train']['loss'].append(loss.item())


def evaluate_ft(netWrapper, loader, history_ft, epoch, args):
    if epoch == 0:
        print('\n-------------------------------------- Fine-tuning ----------------------------------------------------')
    print('Evaluating at epoch {}...'.format(epoch))
    torch.set_grad_enabled(False)

    # remove previous viz results
    makedirs(args.vis_ft, remove=True)

    # switch to eval mode
    netWrapper.eval()

    # initialize meters
    loss_meter = AverageMeter()
    sdr_mix_meter = AverageMeter()
    sdr_meter = AverageMeter()
    sir_meter = AverageMeter()
    sar_meter = AverageMeter()

    eval_num = 0
    valid_num = 0
    with torch.no_grad():
        for i, batch_data in enumerate(loader):
            eval_num += batch_data['mag_mix'].shape[0]
            err_mask, outputs = netWrapper.forward(batch_data, args)
            loss = err_mask.mean()

            loss_meter.update(loss.item())

            total_batch = (11 * args.dup_validset // args.batch_size_val_test)
            if i == 0 or (i + 1) == (total_batch // 2) or (i + 1) == total_batch:
                print('[Eval] iter {}, loss_mask: {:.4f}'.format(i, loss.item()))

            # calculate metrics
            sdr_mix, sdr, sir, sar, cur_valid_num = calc_metrics(batch_data, outputs, args)
            sdr_mix_meter.update(sdr_mix)
            sdr_meter.update(sdr)
            sir_meter.update(sir)
            sar_meter.update(sar)
            valid_num += cur_valid_num

            # output visualization
            if i == 0:
                output_visuals(batch_data, outputs, args)

    metric_output = '[Eval Summary] Epoch: {}, Loss_Mask: {:.4f}, ' \
                    'SDR_mixture: {:.4f}, SDR: {:.4f}, SIR: {:.4f}, SAR: {:.4f}'.format(
        epoch, loss_meter.average(),
        sdr_mix_meter.average(),
        sdr_meter.average(),
        sir_meter.average(),
        sar_meter.average())
    if valid_num / eval_num < 0.8:
        metric_output += ' ---- Invalid ---- '
    print(metric_output)

    learning_rate = ' lr_sound: {}, lr_frame: {}'.format(args.lr_sound_ft, args.lr_frame_ft)
    with open(args.log_ft, 'a') as F:
        if sdr_meter.average() > args.best_sdr:
            F.write(
                '***************************************************************************************************\n')
            F.write(metric_output + learning_rate + '\n')
            F.write(
                '***************************************************************************************************\n')
        else:
            F.write(metric_output + learning_rate + '\n')

    history_ft['val']['epoch'].append(epoch)
    history_ft['val']['loss'].append(loss_meter.average())
    history_ft['val']['sdr'].append(sdr_meter.average())
    history_ft['val']['sir'].append(sir_meter.average())
    history_ft['val']['sar'].append(sar_meter.average())

    # plot figure
    if epoch > 0:
        print('Plotting figures...')
        plot_loss_metrics(args.vis_ft, history_ft)


def evaluate_testset(netWrapper, loader, epoch, args):
    print('==============================Test at epoch {}=============================='.format(epoch))
    torch.set_grad_enabled(False)

    netWrapper.eval()

    # initialize meters
    loss_meter = AverageMeter()
    sdr_mix_meter = AverageMeter()
    sdr_meter = AverageMeter()
    sir_meter = AverageMeter()
    sar_meter = AverageMeter()

    eval_num = 0
    valid_num = 0
    with torch.no_grad():
        for i, batch_data in enumerate(loader):
            eval_num += batch_data['mag_mix'].shape[0]

            err_mask, outputs = netWrapper.forward(batch_data, args)
            loss = err_mask.mean()

            loss_meter.update(loss.item())

            total_batch = (11 * args.dup_testset // args.batch_size_val_test)
            if i == 0 or (i + 1) == (total_batch // 2) or (i + 1) == total_batch:
                print('[Test] iter {}, loss_mask: {:.4f}'.format(i, loss.item()))

            # calculate metrics
            sdr_mix, sdr, sir, sar, cur_valid_num = calc_metrics(batch_data, outputs, args)
            sdr_mix_meter.update(sdr_mix)
            sdr_meter.update(sdr)
            sir_meter.update(sir)
            sar_meter.update(sar)
            valid_num += cur_valid_num

            # # output visualization
            # if i == 0:
            #     output_visuals(batch_data, outputs, args)

    metric_output = '[Test Summary] Epoch: {}, Loss_Mask: {:.4f}, ' \
                    'SDR_mixture: {:.4f}, SDR: {:.4f}, SIR: {:.4f}, SAR: {:.4f}'.format(
        epoch, loss_meter.average(),
        sdr_mix_meter.average(),
        sdr_meter.average(),
        sir_meter.average(),
        sar_meter.average())
    if valid_num / eval_num < 0.8:
        metric_output += ' ---- Invalid ---- '
    print(metric_output)

    with open(args.log_ft, 'a') as F:
        F.write(metric_output + '************************\n')

    print('==============================Test finished!==============================')


def checkpoint(net_frame, net_sound, net_simsiam, args, include_pretrain=False):

    if include_pretrain:
        torch.save(net_frame.state_dict(),
                   '{}/frame_latest.pth'.format(args.ckpt + '/pre-train'))
        torch.save(net_sound.state_dict(),
                   '{}/sound_latest.pth'.format(args.ckpt + '/pre-train'))
        torch.save(net_simsiam.state_dict(),
                   '{}/simsiam_latest.pth'.format(args.ckpt + '/pre-train'))

    else:
        torch.save(net_frame.state_dict(), '{}/frame_best.pth'.format(args.ckpt + '/fine-tune'))
        torch.save(net_sound.state_dict(), '{}/sound_best.pth'.format(args.ckpt + '/fine-tune'))


def create_optimizer(net_frame, net_sound, net_simsiam, args, include_pretrain=False):

    params_conv_pc = [p for p in net_sound.UpConvs.parameters()] + \
                     [p for p in net_sound.DownConvs.parameters()] + \
                     [p for p in net_sound.BNUp.parameters()] + \
                     [p for p in net_sound.BNDown.parameters()] + \
                     [p for p in net_sound.BNUp_step.parameters()] + \
                     [p for p in net_sound.BNDown_step.parameters()]
    params_rate_pc = [p for p in net_sound.a0.parameters()] + \
                     [p for p in net_sound.b0.parameters()]

    if include_pretrain:
        param_groups = [{'params': net_frame.fc.parameters(), 'lr': args.lr_frame_pt},
                        {'params': params_conv_pc, 'lr': args.lr_sound_pt},
                        {'params': params_rate_pc, 'lr': args.lr_sound_pt, 'weight_decay': 0},
                        {'params': net_simsiam.parameters(), 'lr': args.lr_simsiam}]
        return torch.optim.SGD(param_groups, momentum=args.beta1, weight_decay=args.weight_decay)

    else:
        param_groups = [{'params': net_frame.fc.parameters(), 'lr': args.lr_frame_ft},
                        {'params': params_conv_pc, 'lr': args.lr_sound_ft},
                        {'params': params_rate_pc, 'lr': args.lr_sound_ft, 'weight_decay': 0}]
        return torch.optim.AdamW(param_groups, betas=(args.beta1, 0.999))   # default weight_decay is 1e-2


def adjust_learning_rate(optimizer, args, include_pretrain=False):
    if include_pretrain:
        args.lr_sound_pt *= 0.1
        args.lr_frame_pt *= 0.1
        args.lr_simsiam *= 0.1
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1
    else:
        args.lr_sound_ft *= 0.1
        args.lr_frame_ft *= 0.1
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1


if __name__ == '__main__':
    main()
