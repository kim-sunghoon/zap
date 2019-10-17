import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import Config as cfg
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict


class ZAP(nn.Module):
    def __init__(self, planes, mask_type=5):
        super(ZAP, self).__init__()

        self._disabled = False
        self._train = True
        self._stats = []

        self.conv1 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, stride=1, groups=planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, stride=1, groups=planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.mask_type = mask_type
        self.mask = None
        self.mask_c = None
        self.threshold = 1

        self.x_orig = None
        self.x_pred = None

        self.stats = OrderedDict()
        self.stats_hist = OrderedDict()
        self.reset_stats()

    def forward(self, x):
        self.x_orig = x.clone().detach() if self._train is True else None

        if self._disabled is True:
            return x

        # One-time mask creation, assuming the maximal batch size is the one from the configuration
        self._create_mask(x)
        pre_mask = self.mask[0:x.shape[0], 0:x.shape[1], 0:x.shape[2], 0:x.shape[3]]
        pre_mask_c = self.mask_c[0:x.shape[0], 0:x.shape[1], 0:x.shape[2], 0:x.shape[3]]

        # Masked input, i.e., "calculated" only partial ofm elements
        x_pred_mask = torch.mul(x, pre_mask)

        x_pred_mask = self.conv1(x_pred_mask)
        x_pred_mask = self.bn1(x_pred_mask)
        x_pred_mask = F.relu(x_pred_mask)

        x_pred_mask = self.conv2(x_pred_mask)
        x_pred_mask = self.bn2(x_pred_mask)

        x_pred_mask = torch.mul(x_pred_mask, pre_mask_c)

        if self._train:
            x_pred_mask = torch.add(x_pred_mask, pre_mask)
            x_pred_mask = torch.clamp(x_pred_mask, 0, 1)

            self.x_pred = x_pred_mask
        else:
            # pre_mask*100000 is used to differentiate the pre-computed activations from the other activations
            # for analysis. Eventually, x_pred_mask goes through a threshold anyway
            x_pred_mask = torch.add(x_pred_mask, pre_mask*100000)

            self._update_stats_pre_threshold(x, x_pred_mask)
            x_pred_mask = (x_pred_mask > self.threshold).float()
            self._update_stats_post_threshold(x, x_pred_mask, pre_mask)

            self.x_pred = None

        return x * x_pred_mask

    def _update_stats_pre_threshold(self, x, x_pred_mask):
        if cfg.STATS_MASK_VAL_HIST in self._stats:
            # Mask values histogram
            self.stats_hist['M'] += torch.histc(x_pred_mask,
                                                min=cfg.STATS_MASK_VAL_HIST_MIN,
                                                max=cfg.STATS_MASK_VAL_HIST_MAX,
                                                bins=cfg.STATS_MASK_VAL_HIST_BINS).cpu().detach().numpy()

        if cfg.STATS_ERR_TO_TH in self._stats:
            # Original output values
            self.stats_hist['_X_orig_values'] += x.sum().item()

            # For each threshold the numerator is computed
            for i, threshold in enumerate(np.around(np.arange(cfg.STATS_ERR_TO_TH_MIN,
                                                              cfg.STATS_ERR_TO_TH_MAX,
                                                              cfg.STATS_ERR_TO_TH_STEP), 2)):
                x_pred_mask_th = (x_pred_mask > threshold).float()

                # Predicted output values
                # Accumulating the numerator and denominator separately, and dividing at the end
                self.stats_hist['_X_pred_values'][i] += (x * x_pred_mask_th).sum().item()

    def _update_stats_post_threshold(self, x, x_pred_mask, pre_mask):
        if cfg.STATS_GENERAL in self._stats or cfg.STATS_MISPRED_VAL_HIST in self._stats:
            x_ideal_mask = (x > 0).float()  # Notice, this also includes the pre-mask

            # Marking the the pre-mask positions with -100 value
            x_ideal_mask_no_pre_mask = x_ideal_mask.clone().detach()
            x_ideal_mask_no_pre_mask[pre_mask == 1] = -100

            x_pred_mask_no_pre_mask = x_pred_mask.clone().detach()
            x_pred_mask_no_pre_mask[pre_mask == 1] = -100

            # A trick to differentiate the different misprediction and true prediction types
            x_mask_diff = 2 * x_ideal_mask_no_pre_mask - x_pred_mask_no_pre_mask

        if cfg.STATS_GENERAL in self._stats:
            # Total non-zero ofm activations
            self.stats['X_o>0'] += (x > 0).sum().item()
            # Total zero ofm activations
            self.stats['X_o<=0'] += (x <= 0).sum().item()

            # The number of zeroes that are computed by the mask
            self.stats['X_o[I_s]==0'] += ((x_ideal_mask - pre_mask) == -1).sum().item()
            # Remaining zeros, this is the potential
            self.stats['X_orig[I_t]==0'] = self.stats['X_o<=0'] - self.stats['X_o[I_s]==0']

            pre_computed = pre_mask.sum().item()
            # The partially computed ofm activations (|I_s|)
            self.stats['I_s'] += pre_computed
            # The remaining ofm activations to be predicted (|I_t|)
            self.stats['I_t'] += pre_mask.numel() - pre_computed
            # The number of activations to be computed, i.e., that are predicted as non-zero
            self.stats['M==1'] += (x_pred_mask > 0).sum().item() - pre_computed
            # The number of skipped activations (both mispredictions and true-predictions)
            self.stats['M==0'] += (x_pred_mask == 0).sum().item()

            # Activations that are non-zero but were predicted as zeros; may affect accuracy
            self.stats['Miss-!0->0'] += (x_mask_diff == 2).sum().item()
            # True predictions of non-zero activations as non-zero activations, i.e., need to be calculated
            self.stats['Hit-!0->!0'] += (x_mask_diff == 1).sum().item()
            # True predictions of zero activations as zero activations, i.e., no need to be calculated
            self.stats['Hit-0->0'] += (x_mask_diff == 0).sum().item()
            # Activations that are zero but were predicted as non-zero; don't affect accuracy
            self.stats['Miss-0->!0'] += (x_mask_diff == -1).sum().item()

        if cfg.STATS_MISPRED_VAL_HIST in self._stats:
            pred_nonzero_as_zero_values = (x_mask_diff == 2).float() * x
            pred_nonzero_as_zero_values = pred_nonzero_as_zero_values[pred_nonzero_as_zero_values != 0]
            if pred_nonzero_as_zero_values.sum().item() != 0:
                hist = torch.histc(pred_nonzero_as_zero_values, min=cfg.STATS_VAL_HIST_MIN,
                                   max=cfg.STATS_VAL_HIST_MAX,
                                   bins=cfg.STATS_VAL_HIST_BINS).cpu().detach().numpy()
            else:
                hist = np.zeros(int((cfg.STATS_VAL_HIST_MAX - cfg.STATS_VAL_HIST_MIN) / cfg.STATS_VAL_HIST_STEP))
            self.stats_hist['Miss-!0->0'] += hist

    def _create_mask(self, x):
        if self.mask is None:
            # Returns a 2D mask
            self.mask = self._gen_mask(type=self.mask_type, dim=x.shape)
            self.mask_c = torch.abs(self.mask - 1)  # The complementary matrix

            # Add two dimensions
            self.mask.unsqueeze_(0)
            self.mask.unsqueeze_(0)
            self.mask = self.mask.expand(x.shape[0], x.shape[1], -1, -1)

            self.mask_c.unsqueeze_(0)
            self.mask_c.unsqueeze_(0)
            self.mask_c = self.mask_c.expand(x.shape[0], x.shape[1], -1, -1)

    def disable_layer(self):
        self._disabled = True

    def enable_layer(self):
        self._disabled = False

    def disable_grad(self):
        for param in self.parameters():
            param.requires_grad = False

    def enable_grad(self):
        for param in self.parameters():
            param.requires_grad = True

    def status_grad(self):
        status = False

        for param in self.parameters():
            status = status or param.requires_grad

        return status

    def set_pattern(self, type):
        self.mask_type = type
        self.mask, self.mask_c = None, None

    def save_state(self, path):
        state = {'state_dict': self.state_dict(),
                 'mask_type': self.mask_type,
                 'threshold': self.threshold}

        torch.save(state, path)

    def load_state(self, path):
        chkp = torch.load(path)

        self.load_state_dict(chkp['state_dict'])
        self.mask_type = chkp['mask_type']
        self.threshold = chkp['threshold']

    def reset_stats(self):
        self.stats['X_o>0'] = 0
        self.stats['X_o<=0'] = 0
        self.stats['X_o[I_s]==0'] = 0
        self.stats['X_orig[I_t]==0'] = 0
        self.stats['I_s'] = 0
        self.stats['I_t'] = 0
        self.stats['M==1'] = 0
        self.stats['M==0'] = 0
        self.stats['Miss-!0->0'] = 0
        self.stats['Hit-!0->!0'] = 0
        self.stats['Hit-0->0'] = 0
        self.stats['Miss-0->!0'] = 0

        self.stats_hist['_X_orig_values'] = 0
        self.stats_hist['_X_pred_values'] = np.zeros(cfg.STATS_ERR_TO_TH_BINS)

        self.stats_hist['Miss-!0->0'] = np.zeros(cfg.STATS_VAL_HIST_BINS)
        self.stats_hist['Hit-!0->!0'] = np.zeros(cfg.STATS_VAL_HIST_BINS)
        self.stats_hist['M'] = np.zeros(cfg.STATS_MASK_VAL_HIST_BINS)
        self.stats_hist['Error'] = np.zeros(cfg.STATS_ERR_TO_TH_BINS)

    def get_stats(self):
        return list(self.stats.values()), list(self.stats.keys())

    def get_mispred_values_hist(self):
        headers = np.around(np.arange(cfg.STATS_VAL_HIST_MIN,
                                      cfg.STATS_VAL_HIST_MAX,
                                      cfg.STATS_VAL_HIST_STEP), 1).tolist()

        return self.stats_hist['Miss-!0->0'], headers

    def get_mask_values_hist(self):
        headers = np.around(np.arange(cfg.STATS_MASK_VAL_HIST_MIN,
                                      cfg.STATS_MASK_VAL_HIST_MAX,
                                      cfg.STATS_MASK_VAL_HIST_STEP), 3).tolist()

        return self.stats_hist['M'], headers

    def get_err_to_th(self):
        headers = np.around(np.arange(cfg.STATS_ERR_TO_TH_MIN,
                                      cfg.STATS_ERR_TO_TH_MAX,
                                      cfg.STATS_ERR_TO_TH_STEP), 2).tolist()

        if self.stats_hist['_X_orig_values'] == 0:
            return self.stats_hist['_X_pred_values'], headers
        else:
            return 1 - self.stats_hist['_X_pred_values'] / self.stats_hist['_X_orig_values'], headers

    def set_train(self, mode):
        self._train = mode

    def add_stats_gather(self, id):
        if id not in self._stats:
            self._stats.append(id)

    def rm_stats_gather(self):
        if id in self._stats:
            self._stats.remove(id)

    def is_disabled(self):
        return self._disabled

    def reset_layer(self):
        self.apply(self._reset_layer)

    def _reset_layer(self, m):
        if type(m) == nn.Conv2d:
            m.reset_parameters()

    def _gen_mask(self, type: int, dim):
        mask = torch.ones((dim[2], dim[3])).cuda()

        reverse = False
        if type < 5:
            type = 10 - type
            reverse = True

        if type == 5:
            mask[0::2, 1::2] = 0
            mask[1::2, 0::2] = 0
        elif type == 6:
            mask[0::5, 2::5] = 0
            mask[0::5, 4::5] = 0
            mask[1::5, 1::5] = 0
            mask[1::5, 3::5] = 0
            mask[2::5, 0::5] = 0
            mask[2::5, 2::5] = 0
            mask[3::5, 1::5] = 0
            mask[3::5, 4::5] = 0
            mask[4::5, 0::5] = 0
            mask[4::5, 3::5] = 0
        elif type == 7:
            mask[0::3, 2::3] = 0
            mask[1::3, 1::3] = 0
            mask[2::3, 0::3] = 0
        elif type == 8:
            mask[1::2, 1::2] = 0
        elif type == 9:
            mask[1::3, 1::3] = 0
            mask[2::3, 0::3] = 0
        elif type == 10:
            mask[1::3, 1::3] = 0
        else:
            raise NotImplementedError

        if reverse is True:
            mask = torch.abs(mask - 1)

        return mask


class PredNet(nn.Module):
    def __init__(self):
        super().__init__()

        # List of all spatial prediction layers
        self.pred_layers = []   # type: List(ZAP)
        self.disabled_pred_layers = []

    def forward(self, x):
        # Make it abstract
        raise NotImplementedError

    def disable_grad(self):
        for param in self.parameters():
            param.requires_grad = False

    def enable_grad(self, skip_pred=False):
        for param in self.parameters():
            param.requires_grad = True

        if skip_pred is True:
            self.disable_pred_layers_grad()

    def enable_pred_layers_grad(self):
        for layer in self.pred_layers:
            layer.enable_grad()

    def disable_pred_layers_grad(self, indices=None):
        if indices is None:
            for layer in self.pred_layers:
                layer.disable_grad()
        else:
            for idx in indices:
                self.pred_layers[idx].disable_grad()

    def disable_pred_layers(self, indices=None):
        if indices is None:
            for l in self.pred_layers:
                l.disable_layer()
        else:
            for idx in indices:
                self.pred_layers[idx].disable_layer()

    def enable_pred_layers(self):
        for idx, l in enumerate(self.pred_layers):
            if idx in self.disabled_pred_layers:
                continue

            l.enable_layer()

    def rm_all_stats_gather(self):
        for l in self.pred_layers:
            l._stats = []

    def add_stats_gather(self, stats_list):
        if stats_list is None:
            return

        for l in self.pred_layers:
            for stats in stats_list:
                l.add_stats_gather(stats)

    def disable_histograms(self):
        for l in self.pred_layers:
            l._stats_histogram = False

    def print_pred_layers_status(self):
        print("ZAPs status")

        for idx, layer in enumerate(self.pred_layers):
            status = True
            if layer._disabled is True:
                status = False

            grad_status = layer.status_grad()

            print("Feed-forward / grad {}: {} {}".format(idx, status, grad_status))

    def reset_pred_stats(self):
        for pred_layer in self.pred_layers:
            pred_layer.reset_stats()

    def set_pattern(self, pat):
        for l in self.pred_layers:
            l.set_pattern(pat)

    def set_train(self, mode):
        for l in self.pred_layers:
            l.set_train(mode)

    def update_pred_layers_list(self):
        self.apply(self._apply_list_update)

    def _apply_list_update(self, m):
        if type(m) == ZAP:
            self.pred_layers.append(m)
