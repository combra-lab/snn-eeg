import torch
from torch.utils.data import Dataset
import sys
sys.path.append('../')
from offline_train.dataset_utils import generate_left_right_dataset_from_saved_data, get_eeg_2d_map


class EEGDataset2DLeftRight(Dataset):
    """ Reguar Dataset for EEGMMIDB data """
    def __init__(self, base_route, subject_id_list, start_ts, end_ts, window_ts, overlap_ts,
                 use_imagery, use_no_movement, transform=None):
        """
        :param base_route: base route to EEGMMIDB data
        :param subject_id_list: list of subjects
        :param start_ts: start timestep for each slice for epoch
        :param end_ts: maximum end timestep for each slice for epoch
        :param window_ts: window timestep for each epoch
        :param overlap_ts: overlap timestep between two window
        :param use_imagery: if true use imagery data instead of movement data
        :param use_no_movement: if true use no movement as class 0 in movement data
        :param transform: optional transform to be applied on a sample
        """
        self.data, self.label, self.epoch_ts, self.epoch_subjects = generate_left_right_dataset_from_saved_data(
            base_route, subject_id_list, start_ts, end_ts, window_ts, overlap_ts,
            use_imagery, use_no_movement, get_eeg_2d_map(), (10, 11)
        )
        self.transform = transform

    def __len__(self):
        """
        :return: length of the entire dataset
        """
        return self.data.shape[0]

    def __getitem__(self, item):
        """
        Get one entry of data by item
        :param item: index of data
        :return: data with label
        """
        item_label = self.label[item]
        item_data = self.data[item, :, :].reshape(10, 11, -1, 1)
        if self.transform:
            item_data = self.transform(item_data)
        item_data_w_label = [item_data, item_label]
        return item_data_w_label


class ToTensor(object):
    """ Transformation to convert ndarray to pytorch tensor"""
    def __call__(self, sample):
        """
        :param sample: ndarray
        :return: pytorch tensor
        """
        sample = sample.transpose((3, 0, 1, 2))
        sample = torch.from_numpy(sample).float()
        return sample


if __name__ == '__main__':
    ds_params = {"base_route": "../data/eegmmidb_slice_norm/",
                 # "subject_id_list": [i + 1 for i in range(109) if not (i + 1 in [88, 89, 92, 100, 104, 106])],
                 "subject_id_list": [1],
                 "start_ts": 0,
                 "end_ts": 161,
                 "window_ts": 160,
                 "overlap_ts": 0,
                 "use_imagery": False,
                 "use_no_movement": False}
    ds = EEGDataset2DLeftRight(**ds_params)
    print(len(ds))
    eeg_data = ds[0]
    print(eeg_data[0].shape, eeg_data[1])
    print(eeg_data[0][3, 2, :, :])
