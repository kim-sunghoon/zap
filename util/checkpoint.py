import collections
import os


def get_chkp_files(pred_folder):
    dir_files = []
    for dir in pred_folder:
        dir_files_tmp = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]
        for i, filename in enumerate(dir_files_tmp):
            dir_files_tmp[i] = '{}/{}'.format(dir, filename)
        dir_files.extend(dir_files_tmp)

    chkps_dict = {}

    for filename in dir_files:
        filename_split = filename.split('/')[-1].split('_')

        pred_idx = None
        saved_act = None
        mask = None

        for param in filename_split:
            if 'idx' in param:
                pred_idx = int(param.split('-')[1])
            elif 'saved' in param:
                saved_act = float(param[6:-4])
            elif 'mask' in param:
                mask = int(param.split('-')[1])

        if pred_idx is None:
            continue

        if pred_idx not in chkps_dict:
            chkps_dict[pred_idx] = {'saved_act': saved_act, 'mask': mask, 'filename': filename}

    chkps_dict = collections.OrderedDict(sorted(chkps_dict.items()))

    return chkps_dict
