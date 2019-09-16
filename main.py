import torch
import argparse
import util.checkpoint as checkpoint
import Config as cfg
import os
import numpy as np
from NeuralNet import NeuralNet


parser = argparse.ArgumentParser(description='Thanks for Nothing: Predicting Zero-Valued Activations with '
                                             'Lightweight Convolutional Neural Networks\n'
                                             'Gil Shomron, gilsho@campus.technion.ac.il',
                                 formatter_class=argparse.RawTextHelpFormatter)

model_names = ['alexnet-cifar100', 'alexnet-imagenet',
               'vgg16-imagenet',
               'resnet18-imagenet']

parser.add_argument('-a', '--arch', metavar='ARCH', choices=model_names,
                    help='model architectures and datasets:\n ' + ' | '.join(model_names))
parser.add_argument('--phase', choices=['TRAIN_ZAP', 'EVAL_MODEL', 'EVAL_ZAP_ISOL', 'EVAL_ZAP_ROC'],
                    help='TRAIN_ZAP: prediction layers training\n'
                         'EVAL_MODEL: model evaluation and retraining\n'
                         'EVAL_ZAP_ISOL: load prediction layer checkpoints, test in isolation, and present statistics\n'
                         'EVAL_ZAP_ROC: entire model ROC curve')
parser.add_argument('--th_list', nargs='+', default=None,
                    help='thresholds list, relevant only to EVAL_MODEL')
parser.add_argument('--pred_list', nargs='+', default=None,
                    help='ZAPs to train, relevant only for TRAIN_ZAP phase')
parser.add_argument('--mask_list', nargs='+', default=None,
                    help='masks to use')
parser.add_argument('--epochs', default=5, type=int,
                    help='number of epochs for ZAP training or fine-tuning, phase-dependent (default: 5)')
parser.add_argument('--batch_size', default=128, type=int,
                    help='batch size')
parser.add_argument('--lr', default=0.0001, type=float,
                    help='learning rate (default: 0.0001)')
parser.add_argument('--wd', default=0.0005, type=float,
                    help='weight decay (default: 0.0005)')
parser.add_argument('--max_samples', default=None, type=int,
                    help='maximum samples from training set (default: max available)')
parser.add_argument('--no_retrain', action='store_true',
                    help='skip retraining, relevant only for EVAL_MODEL phase (default: False)')
parser.add_argument('--gpu', nargs='+', default=None,
                    help='GPU to run on (default: 0)')


def train_pred_layers(train_gen, test_gen, arch, dataset, model_chkp=None, mask_list=None, epochs=5, pred_list=None):
    """
    Train ZAPs of a pretrained model using the Adam optimizer.
    Each ZAP is trained in isolation.
    :param train_gen: the training set used to train the predictors
    :param test_gen: the test set to evaluate the predictor performance
    :param arch: a string that represents the model architecture, e.g., 'alexnet'
    :param dataset: a string that represents the dataset, e.g., 'cifar100'
    :param model_chkp: a model checkpoint path to be loaded (default: None)
    :param mask_list: specific mask list to train with (default: [6, 5, 4, 3])
    :param epochs: number of epochs to train each ZAP (default: 5)
    :param pred_list: specific prediction layers to train (default: all)
    """
    # Set default masks values
    if mask_list is None:
        mask_list = [6, 5, 4, 3]

    for mask in mask_list:
        cfg.LOG.start_new_log(name='{}-{}_zap-train_mask-{}'.format(arch, dataset, mask))

        mask = int(mask)
        nn = NeuralNet(arch, dataset, model_chkp=model_chkp)

        for pred_idx, pred_layer in enumerate(nn.model.pred_layers):
            if pred_list is not None:
                # Skip ZAPs that are not in the list
                if str(pred_idx) not in pred_list:
                    continue

            cfg.LOG.write_title('ZAP #{}'.format(pred_idx), pad_width=50, pad_symbol='=')
            pred_layer.set_pattern(mask)
            # Threshold is set here for test purposes only, i.e., it does not affect the training process
            pred_layer.threshold = 0.0
            nn.train_pred(train_gen, test_gen, epochs, pred_idx=pred_idx, lr=0.01)

        cfg.LOG.close_log()
        nn = None
        torch.cuda.empty_cache()


def eval_model(train_gen, test_gen, arch, dataset, pred_chkps_dict, model_chkp=None, epochs=5, lr=0.0001, wd=0.0005,
               th_list=None, mask_list=None, fine_tune=True):
    """
    Evaluate and fine-tune an entire model with ZAPs.
    :param train_gen: the training set used to train the predictors
    :param test_gen: the test set to evaluate the predictor performance
    :param arch: a string that represents the model architecture, e.g., 'alexnet'
    :param dataset: a string that represents the dataset, e.g., 'cifar100'
    :param pred_chkps_dict: ZAP checkpoint paths per mask
    :param model_chkp: a model checkpoint path to be loaded (default: None)
    :param epochs: number of epochs to fine-tune the network (default: 5)
    :param lr: fine-tuning learning rate (default: 0.0001)
    :param wd: fine-tuning weight decay (default: 0.0005)
    :param th_list: list of thresholds to evaluate (default: [0, 0.1, 0.2, 0.3, 0.4, 0.5])
    :param fine_tune: continue to fine-tuning after evaluation (default: True)
    :param mask_list: specific mask list (default: everything in pred_chkps_dict)
    """
    # Set default threshold values
    if th_list is None:
        th_list = np.around(np.arange(0, 0.52, 0.1), 2)

    for mask, chkp in pred_chkps_dict.items():
        if mask_list is not None:
            if str(mask) not in mask_list:
                continue

        mask = int(mask)
        pred_chkps = checkpoint.get_chkp_files([chkp])

        for threshold in th_list:
            cfg.LOG.start_new_log(name='{}-{}_full-model_mask-{}_th-{}'.format(arch, dataset, mask, threshold))
            cfg.LOG.write_title("Mask={}".format(mask), pad_width=40, pad_symbol='=')

            nn = NeuralNet(arch, dataset, model_chkp=model_chkp)

            for pred_idx, values in pred_chkps.items():
                nn.load_state_pred(values['filename'])

            cfg.LOG.write_title("Threshold={}".format(threshold), pad_width=40, pad_symbol='=')

            for pred_layer in nn.model.pred_layers:
                pred_layer.threshold = round(float(threshold), 2)

            # Disable first layer in VGG-16 (the other models just don't have ZAP in the first layer)
            if arch == 'vgg16':
                cfg.LOG.write('Disabled first layer in {}'.format(arch))
                nn.model.disabled_pred_layers.append(0)
                nn.model.pred_layers[0].disable_layer()

            nn.test(test_gen, stats=[cfg.STATS_GENERAL])

            # Recouping BN running var and mean
            cfg.LOG.write('BN epoch')
            nn.next_train_epoch = 0
            nn.best_top1_acc = 0
            nn.model.enable_pred_layers()
            nn.model.set_train(False)
            nn.model.enable_grad()
            nn.model.disable_pred_layers_grad()
            nn.train(train_gen, test_gen, 1, lr=0.0, wd=0.0, stats=[cfg.STATS_GENERAL])

            if fine_tune:
                cfg.LOG.write_title("{}-Epoch Fine-Tuning".format(int(epochs)), pad_width=40, pad_symbol='.')
                nn.train(train_gen, test_gen, epochs, lr=lr, wd=wd, stats=[cfg.STATS_GENERAL])

            cfg.LOG.close_log()
            nn = None
            torch.cuda.empty_cache()


def eval_pred_layers(test_gen, arch, dataset, pred_chkps_db, model_chkp=None, mask_list=None):
    """
    Evaluate ZAPs in isolation.
    :param test_gen: the test set to evaluate the predictor performance
    :param arch: a string that represents the model architecture, e.g., 'alexnet'
    :param dataset: a string that represents the dataset, e.g., 'cifar100'
    :param pred_chkps_db: ZAP checkpoint paths per mask
    :param model_chkp: a model checkpoint path to be loaded (default: None)
    :param mask_list: specific mask list (default: everything in pred_chkps_dict)
    """
    for mask, chkp in pred_chkps_db.items():
        if mask_list is not None:
            if str(mask) not in mask_list:
                continue

        mask = int(mask)
        pred_chkps = checkpoint.get_chkp_files([chkp])

        cfg.LOG.start_new_log(name='{}-{}_zap-analysis_mask-{}'.format(arch, dataset, mask))
        cfg.LOG.write_title("Mask={}".format(mask), pad_width=40, pad_symbol='=')

        nn = NeuralNet(arch, dataset, model_chkp=model_chkp)

        for pred_idx, values in pred_chkps.items():
            cfg.LOG.write_title('ZAP #{}'.format(pred_idx), pad_width=50, pad_symbol='=')

            # Load checkpoint
            nn.load_state_pred(values['filename'])
            # Disable all prediction layers
            nn.model.disable_pred_layers()
            # Enable only the current layer
            nn.model.pred_layers[pred_idx].enable_layer()
            # Set threshold (doesn't really matter which)
            nn.model.pred_layers[pred_idx].threshold = 0.0
            # Run test
            nn.test(test_gen, stats=[cfg.STATS_GENERAL, cfg.STATS_MASK_VAL_HIST,
                                     cfg.STATS_MISPRED_VAL_HIST, cfg.STATS_ERR_TO_TH], reset_output=False)

        cfg.LOG.close_log()
        nn = None
        torch.cuda.empty_cache()


def eval_roc(train_gen, test_gen, arch, dataset, pred_chkps_dict, model_chkp=None, mask_list=None, recoup_bn=False):
    """
    Evaluate ZAPs ROC.
    :param train_gen: the training set used to train the predictors
    :param test_gen: the test set to evaluate the predictor performance
    :param arch: a string that represents the model architecture, e.g., 'alexnet'
    :param dataset: a string that represents the dataset, e.g., 'cifar100'
    :param model_chkp: a model checkpoint path to be loaded (default: None)
    :param mask_list: specific mask list to train with (default: [6, 5, 4, 3])
    :param recoup_bn: run BN recoup epoch (default: False)
    """
    for mask, chkp in pred_chkps_dict.items():
        if mask_list is not None:
            if str(mask) not in mask_list:
                continue

        mask = int(mask)
        pred_chkps = checkpoint.get_chkp_files([chkp])

        cfg.LOG.start_new_log(name='{}-{}_zap-analysis-roc_mask-{}'.format(arch, dataset, mask))
        cfg.LOG.write_title("Mask={}".format(mask), pad_width=40, pad_symbol='=')

        nn = NeuralNet(arch, dataset, model_chkp=model_chkp)

        for pred_idx, values in pred_chkps.items():
            nn.load_state_pred(values['filename'])

        # Disable first layer in VGG-16 (the other models just don't have ZAP in the first layer)
        if arch == 'vgg16':
            cfg.LOG.write('Disabled first layer in {}'.format(arch))
            nn.model.disabled_pred_layers.append(0)
            nn.model.pred_layers[0].disable_layer()

        for i, threshold in enumerate(np.around(np.arange(cfg.STATS_ROC_MIN, cfg.STATS_ROC_MAX, cfg.STATS_ROC_STEP), 2)):
            cfg.LOG.write_title("Threshold={}".format(threshold), pad_width=40, pad_symbol='=')

            for pred_layer in nn.model.pred_layers:
                pred_layer.threshold = threshold

            if recoup_bn:
                cfg.LOG.write('BN epoch')
                nn.next_train_epoch = 0
                nn.best_top1_acc = 0
                nn.model.enable_pred_layers()
                nn.model.set_train(False)
                nn.model.enable_grad()
                nn.model.disable_pred_layers_grad()
                nn.train(train_gen, test_gen, 1, lr=0.0, wd=0.0, stats=[cfg.STATS_GENERAL, cfg.STATS_ROC])
            else:
                nn.test(test_gen, stats=[cfg.STATS_GENERAL, cfg.STATS_ROC])

        cfg.LOG.close_log()
        nn = None
        torch.cuda.empty_cache()


def main():
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(args.gpu)

    cfg.BATCH_SIZE = args.batch_size

    arch = args.arch.split('-')[0]
    dataset = args.arch.split('-')[1]

    model_chkp = cfg.get_model_chkp(arch, dataset)
    dataset_ = cfg.get_dataset(dataset)

    test_gen, _ = dataset_.testset(batch_size=args.batch_size)
    (train_gen, _), (_, _) = dataset_.trainset(batch_size=args.batch_size, max_samples=args.max_samples)

    # ZAPs training
    if args.phase == 'TRAIN_ZAP':
        train_pred_layers(train_gen, test_gen, arch, dataset, model_chkp=model_chkp, epochs=args.epochs,
                          mask_list=args.mask_list, pred_list=args.pred_list)

    # Model evaluation and fine-tuning
    elif args.phase == 'EVAL_MODEL':
        chkps = cfg.get_chkps_path(arch, dataset)
        retrain = not args.no_retrain
        eval_model(train_gen, test_gen, arch, dataset, chkps, model_chkp=model_chkp, epochs=args.epochs,
                   lr=args.lr, wd=args.wd, mask_list=args.mask_list, fine_tune=retrain, th_list=args.th_list)

    # Evaluate each ZAP in isolation
    elif args.phase == 'EVAL_ZAP_ISOL':
        chkps = cfg.get_chkps_path(arch, dataset)
        eval_pred_layers(test_gen, arch, dataset, chkps, model_chkp=model_chkp, mask_list=args.mask_list)

    # Evaluate all ZAPs ROC curve
    elif args.phase == 'EVAL_ZAP_ROC':
        chkps = cfg.get_chkps_path(arch, dataset)
        eval_roc(train_gen, test_gen, arch, dataset, chkps, model_chkp=model_chkp, mask_list=args.mask_list)

    return


if __name__ == '__main__':
    main()
