import math
import argparse
import time
import os
import sys
import random
import traceback
from collections import OrderedDict, defaultdict
from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchnet.meter as tnt
script_dir = os.path.dirname(__file__)
module_path = os.path.abspath(os.path.join(script_dir, '..', '..'))
try:
    import distiller
except ImportError:
    sys.path.append(module_path)
    import distiller
import apputils
from distiller.data_loggers import *
import distiller.quantization as quantization
from models import ALL_MODEL_NAMES, create_model

sys.path.append('/host/model_compression/distiller/examples/style_transfer_compression/network')
from utils import *
from neural_style import *

# Logger handle
msglogger = None


def float_range(val_str):
    val = float(val_str)
    if val < 0 or val >= 1:
        raise argparse.ArgumentTypeError('Must be >= 0 and < 1 (received {0})'.format(val_str))
    return val


parser = argparse.ArgumentParser(description='Distiller image classification model compression')
parser.add_argument('--dataset', metavar='DIR', help='path to dataset')
#parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
#                    choices=ALL_MODEL_NAMES,
#                    help='model architecture: ' +
#                    ' | '.join(ALL_MODEL_NAMES) +
#                    ' (default: resnet18)')
#parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
#                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
#parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
#                    help='momentum')
#parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
#                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', default='', type=str, metavar='PretrainedPath',
                    help='use pre-trained model')
parser.add_argument('--act-stats', dest='activation_stats', choices=["train", "valid", "test"], default=None,
                    help='collect activation statistics (WARNING: this slows down training)')
parser.add_argument('--masks-sparsity', dest='masks_sparsity', action='store_true', default=False,
                    help='print masks sparsity table at end of each epoch')
parser.add_argument('--param-hist', dest='log_params_histograms', action='store_true', default=False,
                    help='log the paramter tensors histograms to file (WARNING: this can use significant disk space)')
SUMMARY_CHOICES = ['sparsity', 'compute', 'model', 'modules', 'png', 'png_w_params', 'onnx']
parser.add_argument('--summary', type=str, choices=SUMMARY_CHOICES,
                    help='print a summary of the model, and exit - options: ' +
                    ' | '.join(SUMMARY_CHOICES))
parser.add_argument('--compress', dest='compress', type=str, nargs='?', action='store',
                    help='configuration file for pruning the model (default is to use hard-coded schedule)')
parser.add_argument('--sense', dest='sensitivity', choices=['element', 'filter', 'channel'],
                    help='test the sensitivity of layers to pruning')
parser.add_argument('--extras', default=None, type=str,
                    help='file with extra configuration information')
parser.add_argument('--deterministic', '--det', action='store_true',
                    help='Ensure deterministic execution for re-producible results.')
parser.add_argument('--gpus', metavar='DEV_ID', default=None,
                    help='Comma-separated list of GPU device IDs to be used (default is to use all available devices)')
parser.add_argument('--name', '-n', metavar='NAME', default=None, help='Experiment name')
parser.add_argument('--out-dir', '-o', dest='output_dir', default='logs', help='Path to dump logs and checkpoints')
parser.add_argument('--validation-size', '--vs', type=float_range, default=0.1,
                    help='Portion of training dataset to set aside for validation')
parser.add_argument('--adc', dest='ADC', action='store_true', help='temp HACK')
parser.add_argument('--adc-params', dest='ADC_params', default=None, help='temp HACK')
parser.add_argument('--confusion', dest='display_confusion', default=False, action='store_true',
                    help='Display the confusion matrix')
parser.add_argument('--earlyexit_lossweights', type=float, nargs='*', dest='earlyexit_lossweights', default=None,
                    help='List of loss weights for early exits (e.g. --lossweights 0.1 0.3)')
parser.add_argument('--earlyexit_thresholds', type=float, nargs='*', dest='earlyexit_thresholds', default=None,
                    help='List of EarlyExit thresholds (e.g. --earlyexit 1.2 0.9)')
parser.add_argument('--num-best-scores', dest='num_best_scores', default=1, type=int,
                    help='number of best scores to track and report (default: 1)')
parser.add_argument('--load-serialized', dest='load_serialized', action='store_true', default=False,
                    help='Load a model without DataParallel wrapping it')

quant_group = parser.add_argument_group('Arguments controlling quantization at evaluation time'
                                        '("post-training quantization)')
quant_group.add_argument('--quantize-eval', '--qe', action='store_true',
                         help='Apply linear-symmetric quantization to model before evaluation. Applicable only if'
                              '--evaluate is also set')
quant_group.add_argument('--qe-bits-acts', '--qeba', type=int, default=8, metavar='NUM_BITS',
                         help='Number of bits for quantization of activations')
quant_group.add_argument('--qe-bits-wts', '--qebw', type=int, default=8, metavar='NUM_BITS',
                         help='Number of bits for quantization of weights')
quant_group.add_argument('--qe-bits-accum', type=int, default=32, metavar='NUM_BITS',
                         help='Number of bits for quantization of the accumulator')
quant_group.add_argument('--qe-clip-acts', '--qeca', action='store_true',
                         help='Enable clipping of activations using max-abs-value averaging over batch')
quant_group.add_argument('--qe-no-clip-layers', '--qencl', type=str, nargs='+', metavar='LAYER_NAME', default=[],
                         help='List of fully-qualified layer names for which not to clip activations. Applicable'
                              'only if --qe-clip-acts is also set')
parser.add_argument("--image-size", type=int, default=256,
                    help="size of training images, default is 256 X 256")
parser.add_argument("--style-size", type=int, default=None,
                    help="size of style-image, default is the original size of style image")
parser.add_argument('--content-weight', default=1e5, type=float,
                    metavar='CW', help='Content Weights')
parser.add_argument('--style-weight', default=1e10, type=float,
                    metavar='SW', help='Style Weights')
parser.add_argument('--style-image', default='', type=str, metavar='StyleImagePath',
                    help='Style Image Path')
parser.add_argument('--cuda', default=1, type=int,
                    metavar='CUDA', help='specify which cuda to use (default: 1)')

distiller.knowledge_distillation.add_distillation_args(parser, ALL_MODEL_NAMES, True)


def check_pytorch_version():
    if torch.__version__ < '0.4.0':
        print("\nNOTICE:")
        print("The Distiller \'master\' branch now requires at least PyTorch version 0.4.0 due to "
              "PyTorch API changes which are not backward-compatible.\n"
              "Please install PyTorch 0.4.0 or its derivative.\n"
              "If you are using a virtual environment, do not forget to update it:\n"
              "  1. Deactivate the old environment\n"
              "  2. Install the new environment\n"
              "  3. Activate the new environment")
        exit(1)


def create_activation_stats_collectors(model, collection_phase):
    """Create objects that collect activation statistics.

    This is a utility function that creates two collectors:
    1. Fine-grade sparsity levels of the activations
    2. L1-magnitude of each of the activation channels

    Args:
        model - the model on which we want to collect statistics
        phase - the statistics collection phase which is either "train" (for training),
                or "valid" (for validation)

    WARNING! Enabling activation statsitics collection will significantly slow down training!
    """
    class missingdict(dict):
        """This is a little trick to prevent KeyError"""
        def __missing__(self, key):
            return None  # note, does *not* set self[key] - we don't want defaultdict's behavior

    distiller.utils.assign_layer_fq_names(model)

    activations_collectors = {"train": missingdict(), "valid": missingdict(), "test": missingdict()}
    if collection_phase is None:
        return activations_collectors
    collectors = missingdict()
    collectors["sparsity"] = SummaryActivationStatsCollector(model, "sparsity", distiller.utils.sparsity)
    collectors["l1_channels"] = SummaryActivationStatsCollector(model, "l1_channels",
                                                                distiller.utils.activation_channels_l1)
    collectors["apoz_channels"] = SummaryActivationStatsCollector(model, "apoz_channels",
                                                                  distiller.utils.activation_channels_apoz)
    collectors["records"] = RecordsActivationStatsCollector(model, classes=[torch.nn.Conv2d])
    activations_collectors[collection_phase] = collectors
    return activations_collectors


def save_collectors_data(collectors, directory):
    """Utility function that saves all activation statistics to Excel workbooks
    """
    for name, collector in collectors.items():
        collector.to_xlsx(os.path.join(directory, name))


def main():
    global msglogger
    check_pytorch_version()
    args = parser.parse_args()
    print(args)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    msglogger = apputils.config_pylogger(os.path.join(script_dir, 'logging.conf'), args.name, args.output_dir)

    # Log various details about the execution environment.  It is sometimes useful
    # to refer to past experiment executions and this information may be useful.
    apputils.log_execution_env_state(sys.argv, gitroot=module_path)
    msglogger.debug("Distiller: %s", distiller.__version__)

    start_epoch = 0
    best_epochs = [distiller.MutableNamedTuple({'epoch': 0, 'loss': float("inf"), 'sparsity': 0})
                   for i in range(args.num_best_scores)]

    if args.deterministic:
        # Experiment reproducibility is sometimes important.  Pete Warden expounded about this
        # in his blog: https://petewarden.com/2018/03/19/the-machine-learning-reproducibility-crisis/
        # In Pytorch, support for deterministic execution is still a bit clunky.
        if args.workers > 1:
            msglogger.error('ERROR: Setting --deterministic requires setting --workers/-j to 0 or 1')
            exit(1)
        # Use a well-known seed, for repeatability of experiments
        torch.manual_seed(0)
        random.seed(0)
        np.random.seed(0)
        cudnn.deterministic = True
    else:
        # This issue: https://github.com/pytorch/pytorch/issues/3659
        # Implies that cudnn.benchmark should respect cudnn.deterministic, but empirically we see that
        # results are not re-produced when benchmark is set. So enabling only if deterministic mode disabled.
        cudnn.benchmark = True

    # if args.gpus is not None:
    #     try:
    #         args.gpus = [int(s) for s in args.gpus.split(',')]
    #     except ValueError:
    #         msglogger.error('ERROR: Argument --gpus must be a comma-separated list of integers only')
    #         exit(1)
    #     available_gpus = torch.cuda.device_count()
    #     for dev_id in args.gpus:
    #         if dev_id >= available_gpus:
    #             msglogger.error('ERROR: GPU device ID {0} requested, but only {1} devices available'
    #                             .format(dev_id, available_gpus))
    #             exit(1)
    #     # Set default device in case the first one on the list != 0
    #     torch.cuda.set_device(args.gpus[0])
    #
    # # Infer the dataset from the model name
    # args.dataset = 'cifar10' if 'cifar' in args.arch else 'imagenet'
    # args.num_classes = 10 if args.dataset == 'cifar10' else 1000

    if args.earlyexit_thresholds:
        args.num_exits = len(args.earlyexit_thresholds) + 1
        args.loss_exits = [0] * args.num_exits
        args.losses_exits = []
        args.exiterrors = []

    # Create the model
    model = TransformerNet()
    if args.cuda:
        device = torch.device("cuda:{}".format(args.cuda - 1))
    else:
        device = torch.device("cpu")
    model.to(device)
    vgg = Vgg16(requires_grad=False).to(device)
    style_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    style = utils.load_image(args.style_image, size=args.style_size)
    style = style_transform(style)
    style = style.repeat(args.batch_size, 1, 1, 1).to(device)
    if args.pretrained:
        resumed_state_dict = torch.load(args.pretrained)
        # remove saved deprecated running_* keys in InstanceNorm from the checkpoint
        for k in list(resumed_state_dict.keys()):
            if re.search(r'in\d+\.running_(mean|var)$', k):
                del resumed_state_dict[k]
        model.load_state_dict(resumed_state_dict)
        msglogger.info('Loaded pretrained model from %s\n', args.pretrained)

    compression_scheduler = None
    # Create a couple of logging backends.  TensorBoardLogger writes log files in a format
    # that can be read by Google's Tensor Board.  PythonLogger writes to the Python logger.
    tflogger = TensorBoardLogger(msglogger.logdir)
    pylogger = PythonLogger(msglogger)

    # capture thresholds for early-exit training
    if args.earlyexit_thresholds:
        msglogger.info('=> using early-exit threshold values of %s', args.earlyexit_thresholds)

    # We can optionally resume from a checkpoint
    if args.resume:
        model, compression_scheduler, start_epoch = apputils.load_checkpoint(
            model, chkpt_file=args.resume)

    # Define loss function (criterion) and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = Adam(model.parameters(), args.lr)
    msglogger.info('Optimizer Type: %s', type(optimizer))
    msglogger.info('Optimizer Args: %s', optimizer.defaults)

    if args.ADC:
        return automated_deep_compression(model, criterion, pylogger, args)

    # This sample application can be invoked to produce various summary reports.
    if args.summary:
        return summarize_model(model, args.dataset, which_summary=args.summary)

    # Load the datasets: the dataset to load is inferred from the model name passed
    # in args.arch.  The default dataset is ImageNet, but if args.arch contains the
    # substring "_cifar", then cifar10 is used.
    transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    train_dataset = datasets.ImageFolder(args.dataset, transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)
    msglogger.info('Dataset sizes:\n\ttraining=%d\n',
                   len(train_loader.sampler))

    activations_collectors = create_activation_stats_collectors(model, collection_phase=args.activation_stats)

    if args.sensitivity is not None:
        return sensitivity_analysis(model, criterion, test_loader, pylogger, args)

    if args.evaluate:
        return evaluate_model(model, criterion, test_loader, pylogger, activations_collectors, args)

    if args.compress:
        # The main use-case for this sample application is CNN compression. Compression
        # requires a compression schedule configuration file in YAML.
        compression_scheduler = distiller.file_config(model, optimizer, args.compress)
        # Model is re-transferred to GPU in case parameters were added (e.g. PACTQuantizer)
        model.to(device)
    else:
        compression_scheduler = distiller.CompressionScheduler(model)

    args.kd_policy = None
    if args.kd_teacher:
        teacher = create_model(args.kd_pretrained, args.dataset, args.kd_teacher, device_ids=args.gpus)
        if args.kd_resume:
            teacher, _, _ = apputils.load_checkpoint(teacher, chkpt_file=args.kd_resume)
        dlw = distiller.DistillationLossWeights(args.kd_distill_wt, args.kd_student_wt, args.kd_teacher_wt)
        args.kd_policy = distiller.KnowledgeDistillationPolicy(model, teacher, args.kd_temp, dlw)
        compression_scheduler.add_policy(args.kd_policy, starting_epoch=args.kd_start_epoch, ending_epoch=args.epochs,
                                         frequency=1)

        msglogger.info('\nStudent-Teacher knowledge distillation enabled:')
        msglogger.info('\tTeacher Model: %s', args.kd_teacher)
        msglogger.info('\tTemperature: %s', args.kd_temp)
        msglogger.info('\tLoss Weights (distillation | student | teacher): %s',
                       ' | '.join(['{:.2f}'.format(val) for val in dlw]))
        msglogger.info('\tStarting from Epoch: %s', args.kd_start_epoch)

    for epoch in range(start_epoch, start_epoch + args.epochs):
        # This is the main training loop.
        msglogger.info('\n')
        if compression_scheduler:
            compression_scheduler.on_epoch_begin(epoch)

        # Train for one epoch
        with collectors_context(activations_collectors["train"]) as collectors:
            train(train_loader, model, criterion, optimizer, vgg, epoch, compression_scheduler, [tflogger, pylogger],
                  args.print_freq, style, args.content_weight, args.style_weight, device)
            distiller.log_weights_sparsity(model, epoch, loggers=[tflogger, pylogger])
            distiller.log_activation_statsitics(epoch, "train", loggers=[tflogger],
                                                collector=collectors["sparsity"])
            if args.masks_sparsity:
                msglogger.info(distiller.masks_sparsity_tbl_summary(model, compression_scheduler))

        # evaluate on validation set
        with collectors_context(activations_collectors["valid"]) as collectors:
            top1, top5, vloss = validate(train_loader, model, criterion, vgg, [pylogger], args, style, device, epoch)
            distiller.log_activation_statsitics(epoch, "valid", loggers=[tflogger],
                                                collector=collectors["sparsity"])
            save_collectors_data(collectors, msglogger.logdir)

        if compression_scheduler:
            compression_scheduler.on_epoch_end(epoch, optimizer)

        # remember best top1 and save checkpoint
        #sparsity = distiller.model_sparsity(model)
        is_best = vloss < best_epochs[0].loss
        if is_best:
            best_epochs[0].epoch = epoch
            best_epochs[0].loss = vloss
            best_epochs = sorted(best_epochs, key=lambda score: score.loss, reverse=True)
        for score in reversed(best_epochs):
            if score.loss > 0:
                msglogger.info('==> Best Loss: %.3f on Epoch: %d', score.loss, score.epoch)
        apputils.save_checkpoint(epoch, None, model, optimizer, compression_scheduler,
                                 best_epochs[0].loss, is_best, args.name, msglogger.logdir)

OVERALL_LOSS_KEY = 'Overall Loss'
OBJECTIVE_LOSS_KEY = 'Objective Loss'


def train(train_loader, model, criterion, optimizer, vgg, epoch, compression_scheduler, loggers,
          print_freq, style, content_weight, style_weight, device):
    #     np.random.seed(args.seed)
    #     torch.manual_seed(args.seed)
    """Training loop for one epoch."""
    losses = OrderedDict([(OVERALL_LOSS_KEY, tnt.AverageValueMeter()),
                          (OBJECTIVE_LOSS_KEY, tnt.AverageValueMeter())])

    batch_time = tnt.AverageValueMeter()

    total_samples = len(train_loader.sampler)
    batch_size = train_loader.batch_size
    steps_per_epoch = math.ceil(total_samples / batch_size)
    msglogger.info('Training epoch: %d samples (%d per mini-batch)', total_samples, batch_size)

    features_style = vgg(utils.normalize_batch(style))
    gram_style = [utils.gram_matrix(y) for y in features_style]

    # Switch to train mode
    model.train()
    end = time.time()

    for train_step, (x, _) in enumerate(train_loader):
        # Execute the forward phase, compute the output and measure loss
        if compression_scheduler:
            compression_scheduler.on_minibatch_begin(epoch, train_step, steps_per_epoch, optimizer)

        x = x.to(device)
        y = model(x)

        y = utils.normalize_batch(y)
        x = utils.normalize_batch(x)

        features_y = vgg(y)
        features_x = vgg(x)

        content_loss = content_weight * criterion(features_y.relu2_2, features_x.relu2_2)

        style_loss = 0.
        for ft_y, gm_s in zip(features_y, gram_style):
            gm_y = utils.gram_matrix(ft_y)
            style_loss += criterion(gm_y, gm_s[:batch_size, :, :])
        style_loss *= style_weight

        loss = content_loss + style_loss

        losses[OBJECTIVE_LOSS_KEY].add(loss.item())

        if compression_scheduler:
            # Before running the backward phase, we allow the scheduler to modify the loss
            # (e.g. add regularization loss)
            agg_loss = compression_scheduler.before_backward_pass(epoch, train_step, steps_per_epoch, loss,
                                                                  optimizer=optimizer, return_loss_components=True)
            loss = agg_loss.overall_loss
            losses[OVERALL_LOSS_KEY].add(loss.item())
            for lc in agg_loss.loss_components:
                if lc.name not in losses:
                    losses[lc.name] = tnt.AverageValueMeter()
                losses[lc.name].add(lc.value.item())

        # Compute the gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if compression_scheduler:
            compression_scheduler.on_minibatch_end(epoch, train_step, steps_per_epoch, optimizer)

        # measure elapsed time
        batch_time.add(time.time() - end)
        steps_completed = (train_step+1)

        if steps_completed % print_freq == 0:
            stats_dict = OrderedDict()
            for loss_name, meter in losses.items():
                stats_dict[loss_name] = meter.mean
            # stats_dict.update(errs)
            stats_dict['LR'] = optimizer.param_groups[0]['lr']
            stats_dict['Time'] = batch_time.mean
            stats = ('Peformance/Training/', stats_dict)

            # params = model.named_parameters() if args.log_params_histograms else None
            params = None
            distiller.log_training_progress(stats,
                                            params,
                                            epoch, steps_completed,
                                            steps_per_epoch, print_freq,
                                            loggers)

        end = time.time()


def validate(val_loader, model, criterion, vgg, loggers, args, style, device, epoch=-1):
    """Model validation"""
    if epoch > -1:
        msglogger.info('--- validate (epoch=%d)-----------', epoch)
    else:
        msglogger.info('--- validate ---------------------')
    return _validate(val_loader, model, criterion, vgg, loggers, args, style, device, epoch)


def _validate(data_loader, model, criterion, vgg, loggers, args, style, device, epoch=-1):
    """Execute the validation/test loop."""
    losses = {'objective_loss': tnt.AverageValueMeter()}
    # classerr = tnt.ClassErrorMeter(accuracy=True, topk=(1, 5))

    if args.earlyexit_thresholds:
        # for Early Exit, we have a list of errors and losses for each of the exits.
        args.exiterrors = []
        args.losses_exits = []
        for exitnum in range(args.num_exits):
            # args.exiterrors.append(tnt.ClassErrorMeter(accuracy=True, topk=(1, 5)))
            args.losses_exits.append(tnt.AverageValueMeter())
        args.exit_taken = [0] * args.num_exits

    batch_time = tnt.AverageValueMeter()
    total_samples = len(data_loader.sampler)
    batch_size = data_loader.batch_size
    if args.display_confusion:
        confusion = tnt.ConfusionMeter(args.num_classes)
    total_steps = total_samples / batch_size
    msglogger.info('%d samples (%d per mini-batch)', total_samples, batch_size)

    steps_per_epoch = math.ceil(total_samples / batch_size)
    msglogger.info('Training epoch: %d samples (%d per mini-batch)', total_samples, batch_size)

    features_style = vgg(utils.normalize_batch(style))
    gram_style = [utils.gram_matrix(y) for y in features_style]

    # Switch to evaluation mode
    model.eval()

    end = time.time()
    for validation_step, (x, target) in enumerate(data_loader):
        with torch.no_grad():
            if not args.earlyexit_thresholds:
                x = x.to(device)
                y = model(x)

                y = utils.normalize_batch(y)
                x = utils.normalize_batch(x)

                features_y = vgg(y)
                features_x = vgg(x)

                content_loss = args.content_weight * criterion(features_y.relu2_2, features_x.relu2_2)

                style_loss = 0.
                for ft_y, gm_s in zip(features_y, gram_style):
                    gm_y = utils.gram_matrix(ft_y)
                    style_loss += criterion(gm_y, gm_s[:batch_size, :, :])
                style_loss *= args.style_weight

                loss = content_loss + style_loss

                losses['objective_loss'].add(loss.item())
                # classerr.add(output.data, target)
                if args.display_confusion:
                    confusion.add(output.data, target)
            else:
                earlyexit_validate_loss(output, target, criterion, args)

            # measure elapsed time
            batch_time.add(time.time() - end)
            end = time.time()

            steps_completed = (validation_step+1)
            if steps_completed % args.print_freq == 0:
                if not args.earlyexit_thresholds:
                    stats = ('',
                             OrderedDict([('Loss', losses['objective_loss'].mean)]))
                #                             OrderedDict([('Loss', losses['objective_loss'].mean),
                #                                          ('Top1', classerr.value(1)),
                #                                          ('Top5', classerr.value(5))]))
                else:
                    stats_dict = OrderedDict()
                    stats_dict['Test'] = validation_step
                    for exitnum in range(args.num_exits):
                        la_string = 'LossAvg' + str(exitnum)
                        stats_dict[la_string] = args.losses_exits[exitnum].mean
                        # Because of the nature of ClassErrorMeter, if an exit is never taken during the batch,
                        # then accessing the value(k) will cause a divide by zero. So we'll build the OrderedDict
                        # accordingly and we will not print for an exit error when that exit is never taken.
                        if args.exit_taken[exitnum]:
                            t1 = 'Top1_exit' + str(exitnum)
                            t5 = 'Top5_exit' + str(exitnum)
                            stats_dict[t1] = args.exiterrors[exitnum].value(1)
                            stats_dict[t5] = args.exiterrors[exitnum].value(5)
                    stats = ('Performance/Validation/', stats_dict)

                distiller.log_training_progress(stats, None, epoch, steps_completed,
                                                total_steps, args.print_freq, loggers)
    if not args.earlyexit_thresholds:
        msglogger.info('==> Top1: %.3f    Top5: %.3f    Loss: %.3f\n',
                       0, 0, losses['objective_loss'].mean)

        if args.display_confusion:
            msglogger.info('==> Confusion:\n%s\n', str(confusion.value()))
        return 0, 0, losses['objective_loss'].mean
    else:
        total_top1, total_top5, losses_exits_stats = earlyexit_validate_stats(args)
        return total_top1, total_top5, losses_exits_stats[args.num_exits-1]


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n-- KeyboardInterrupt --")
    except Exception as e:
        if msglogger is not None:
            msglogger.error(traceback.format_exc())
        raise
    finally:
        if msglogger is not None:
            msglogger.info('')
            msglogger.info('Log file for this run: ' + os.path.realpath(msglogger.log_filename))
