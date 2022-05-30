import pickle
import numpy as np
import copy


def generate_left_right_dataset_from_saved_data(base_route, subject_id_list, start_ts, end_ts, window_ts, overlap_ts,
                                                use_imagery, use_no_movement, eeg_2d_map, map_2d_shape):
    """
    Only use Left and Right fist data from the pre-processed MMIDB dataset

    :param base_route: base directory to the pre-processed MMIDB dataset
    :param subject_id_list: list of subject to be included in the data
    :param start_ts: start timestep for data slicing
    :param end_ts: end timestep for data slicing
    :param window_ts: data slicing window
    :param overlap_ts: data slicing overlap timestep
    :param use_imagery: if true use imagery data else use movement data
    :param use_no_movement: if true include no movement data as class 0
    :param eeg_2d_map: 2D mapping of EEG
    :param map_2d_shape: 2D shape of EEG
    :return: mmidb_full_epoch_data, mmidb_full_label, mmidb_full_epoch_ts, mmidb_full_subjects
    """
    mmidb_full_epoch_data, mmidb_full_label, mmidb_full_epoch_ts, mmidb_full_subjects = [], [], [], []
    if use_imagery:
        task_list = [4, 8, 12]
    else:
        task_list = [3, 7, 11]
    for subject_id in subject_id_list:
        subject_str = 'S%03d' % subject_id
        for task in task_list:
            task_str = subject_str + 'R%02d' % task
            saved_data_list = pickle.load(open(base_route + subject_str + '/' + task_str + '.p', 'rb'))
            label_list = pickle.load(open(base_route + subject_str + '/' + task_str + '_label.p', 'rb'))
            saved_2d_data_list = transform_slice_raw_data_2_2d(saved_data_list, eeg_2d_map, map_2d_shape)
            epoch_data_list, epoch_label_list, epoch_ts_list = epoch_2d_data_w_label(
                saved_2d_data_list, label_list, start_ts, end_ts, window_ts, overlap_ts)
            if not use_no_movement:
                epoch_label_list = epoch_label_list - 2
                epoch_data_list = epoch_data_list[epoch_label_list >= 0, :, :, :]
                epoch_ts_list = epoch_ts_list[epoch_label_list >= 0, :]
                epoch_label_list = epoch_label_list[epoch_label_list >= 0]
            else:
                epoch_label_list = epoch_label_list - 1
            mmidb_full_epoch_data.append(epoch_data_list)
            mmidb_full_label.extend(epoch_label_list.tolist())
            mmidb_full_epoch_ts.append(epoch_ts_list)
            mmidb_full_subjects.extend([subject_id for _ in epoch_label_list])
    mmidb_full_epoch_data = np.concatenate(mmidb_full_epoch_data, axis=0)
    mmidb_full_label = np.array(mmidb_full_label)
    mmidb_full_epoch_ts = np.concatenate(mmidb_full_epoch_ts, axis=0)
    mmidb_full_subjects = np.array(mmidb_full_subjects)
    return mmidb_full_epoch_data, mmidb_full_label, mmidb_full_epoch_ts, mmidb_full_subjects


def get_eeg_2d_map():
    """
    Generate map for 2D transformation
    :return: eeg_2d_map
    """
    eeg_2d_map = [(3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (3, 7), (3, 8),
                  (4, 2), (4, 3), (4, 4), (4, 5), (4, 6), (4, 7), (4, 8),
                  (5, 2), (5, 3), (5, 4), (5, 5), (5, 6), (5, 7), (5, 8),
                  (0, 4), (0, 5), (0, 6),
                  (1, 3), (1, 4), (1, 5), (1, 6), (1, 7),
                  (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8), (2, 9),
                  (3, 1), (3, 9),
                  (4, 1), (4, 9), (4, 0), (4, 10),
                  (5, 1), (5, 9),
                  (6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (6, 6), (6, 7), (6, 8), (6, 9),
                  (7, 3), (7, 4), (7, 5), (7, 6), (7, 7),
                  (8, 4), (8, 5), (8, 6),
                  (9, 5)]
    return eeg_2d_map


def transform_slice_raw_data_2_2d(mmidb_norm_raw_data_list, eeg_2d_map, map_2d_shape):
    """
    Transform slice raw EEG data to 2D EEG data based on the map

    :param mmidb_norm_raw_data_list: List of normalized raw slice data
    :param eeg_2d_map: map for 2D transformation
    :param map_2d_shape: shape of the 2D map
    :return: mmidb_2d_raw_data_list
    """
    mmidb_2d_raw_data_list = []
    for slice_raw_data in mmidb_norm_raw_data_list:
        channel_num = slice_raw_data.shape[0]
        slice_ts = slice_raw_data.shape[1]
        map_raw_data = np.zeros((map_2d_shape[0], map_2d_shape[1], slice_ts))
        for num in range(channel_num):
            map_raw_data[eeg_2d_map[num][0], eeg_2d_map[num][1], :] = slice_raw_data[num, :]
        mmidb_2d_raw_data_list.append(map_raw_data)
    return mmidb_2d_raw_data_list


def epoch_2d_data_w_label(mmidb_2d_raw_data_list, mmidb_label_list, start_ts, end_ts, window_ts, overlap_ts):
    """
    Epoch the 2D data

    :param mmidb_2d_raw_data_list: list of 2D raw data
    :param mmidb_label_list: list of label for each slice raw data
    :param start_ts: start timestep for each slice for epoch
    :param end_ts: maximum end timestep for each slice for epoch
    :param window_ts: window timestep for each epoch
    :param overlap_ts: overlap timestep between two window
    :return: mmidb_epoch_data, mmidb_epoch_label, mmidb_epoch_ts
    """
    raw_epoch_data_list, raw_epoch_label_list, raw_epoch_ts_list = [], [], []
    for slice_idx, slice_data in enumerate(mmidb_2d_raw_data_list, 0):
        slice_label = mmidb_label_list[slice_idx]
        epoch_start_ts = start_ts
        epoch_end_ts = epoch_start_ts + window_ts
        while epoch_end_ts < slice_data.shape[2] and epoch_end_ts < end_ts:
            raw_epoch_data = copy.deepcopy(slice_data[:, :, epoch_start_ts:epoch_end_ts])
            raw_epoch_data_list.append(raw_epoch_data)
            raw_epoch_label_list.append(slice_label)
            raw_epoch_ts_list.append([slice_idx, epoch_start_ts, epoch_end_ts])
            epoch_start_ts = epoch_end_ts - overlap_ts
            epoch_end_ts = epoch_start_ts + window_ts
    mmidb_epoch_data = np.array(raw_epoch_data_list)
    mmidb_epoch_label = np.array(raw_epoch_label_list)
    mmidb_epoch_ts = np.array(raw_epoch_ts_list)
    return mmidb_epoch_data, mmidb_epoch_label, mmidb_epoch_ts