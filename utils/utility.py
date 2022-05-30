import mne
import numpy as np
import copy
import os
import pickle
from typing import Tuple, List
from random import shuffle, random


def read_raw_data_n_events(base_route, subject_id, task_id):
    # type: (str, int, int) -> Tuple[np.ndarray, np.ndarray]
    """
    Read EEG Raw Data and Events from EDF file

    :param base_route: base route to EEGMMIDB data
    :param subject_id: Subject ID
    :param task_id: Task ID
    :return: mmidb_raw_data, mmidb_events
    """
    subject_str, task_str = 'S%03d' % subject_id, 'R%02d' % task_id
    data_route = base_route + subject_str + '/' + subject_str + task_str + '.edf'
    raw_edf_data = mne.io.read_raw_edf(data_route, preload=True)
    raw_edf_data.filter(0.1, 79., fir_design='firwin')
    mmidb_events, _ = mne.events_from_annotations(raw_edf_data)
    mmidb_raw_data = raw_edf_data.get_data()
    return mmidb_raw_data, mmidb_events


def slice_raw_data_between_events(mmidb_raw_data, mmidb_events):
    # type: (np.ndarray, np.ndarray) -> Tuple[List[np.ndarray], List[int]]
    """
    Slice the data between two events for one trial of EEG data

    :param mmidb_raw_data: One trial of EEG data
    :param mmidb_events: Events for EEG data
    :return: mmidb_raw_data_list, mmidb_label_list
    """
    mmidb_raw_data_list, mmidb_label_list = [], []
    for event_idx in range(mmidb_events.shape[0]):
        mmidb_label_list.append(mmidb_events[event_idx, 2])
        event_start_ts = mmidb_events[event_idx, 0]
        if event_idx == mmidb_events.shape[0] - 1:
            event_end_ts = mmidb_raw_data.shape[1]
        else:
            event_end_ts = mmidb_events[event_idx + 1, 0]
        event_data_slice = copy.deepcopy(mmidb_raw_data[:, event_start_ts:event_end_ts])
        mmidb_raw_data_list.append(event_data_slice)
    return mmidb_raw_data_list, mmidb_label_list


def normalize_slice_raw_data(mmidb_raw_data_list, channel_mean, channel_std):
    # type: (List[np.ndarray], np.ndarray, np.ndarray) -> List[np.ndarray]
    """
    Perform Z-score Norm on slice raw EEG data
    :param mmidb_raw_data_list: List of raw slice data
    :param channel_mean: mean for each channel
    :param channel_std: std for each channel
    :return: mmidb_norm_raw_data_list
    """
    mmidb_norm_raw_data_list = []
    channel_mean, channel_std = channel_mean.reshape(-1, 1), channel_std.reshape(-1, 1)
    for slice_raw_data in mmidb_raw_data_list:
        slice_raw_data = (slice_raw_data - channel_mean) / channel_std
        mmidb_norm_raw_data_list.append(slice_raw_data)
    return mmidb_norm_raw_data_list


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
    # type: (List[np.ndarray], List[Tuple[int, int]], Tuple[int, int]) -> List[np.ndarray]
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
    # type: (List[np.ndarray], List[int], int, int, int, int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]
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
    


def compute_mean_std_for_all_data(base_route, subject_id_list):
    """
    Compute Global Mean and STD for each channel

    :param base_route: base route for EEGMMIDB data
    :param subject_id_list: list of subjects
    :return: channel_mean, channel_std
    """
    raw_data_list = []
    raw_data_overall_length = 0
    for idx, subject_id in enumerate(subject_id_list, 0):
        for task in range(14):
            raw_data, events = read_raw_data_n_events(base_route, subject_id, task + 1)
            raw_data_list.append(raw_data)
            raw_data_overall_length += raw_data.shape[1]
    raw_data_numpy_list = np.zeros((64, raw_data_overall_length))
    append_start = 0
    for raw_data in raw_data_list:
        raw_data_numpy_list[:, append_start:append_start + raw_data.shape[1]] = raw_data
        append_start = append_start + raw_data.shape[1]
    channel_mean = np.mean(raw_data_numpy_list, axis=1)
    channel_std = np.std(raw_data_numpy_list, axis=1)
    channel_max = np.max(raw_data_numpy_list, axis=1)
    channel_min = np.min(raw_data_numpy_list, axis=1)
    return channel_mean, channel_std, channel_max, channel_min


def generate_and_save_all_slice_data(base_route, base_save_route, subject_id_list,
                                     channel_mean, channel_std):
    """
    Slice and Normalize EEG data and save to files

    :param base_route: base route for EEGMMIDB data
    :param base_save_route: base route for saving new data
    :param subject_id_list: list of subjects
    :param channel_mean: mean for each channel
    :param channel_std: std for each channel
    """
    try:
        os.mkdir(base_save_route)
        print("Directory ", base_save_route, " created ...")
    except FileExistsError:
        print("Directory ", base_save_route, " already exists ...")
    for idx, subject_id in enumerate(subject_id_list, 0):
        subject_str = 'S%03d' % subject_id
        try:
            os.mkdir(base_save_route + subject_str + '/')
            print("Directory ", base_save_route + subject_str + '/', " created ...")
        except FileExistsError:
            print("Directory ", base_save_route + subject_str + '/', " already exists ...")
        for task in range(14):
            task_str = subject_str + 'R%02d' % (task + 1)
            raw_data, events = read_raw_data_n_events(base_route, subject_id, task + 1)
            raw_data_list, label_list = slice_raw_data_between_events(raw_data, events)
            norm_raw_data_list = normalize_slice_raw_data(raw_data_list, channel_mean, channel_std)
            pickle.dump(norm_raw_data_list, open(base_save_route + subject_str + '/' + task_str + '.p', 'wb+'))
            pickle.dump(label_list, open(base_save_route + subject_str + '/' + task_str + '_label.p', 'wb+'))
            print("Save Slice Norm Data and Labels ", task_str)


def generate_left_right_dataset_from_saved_data(base_route, subject_id_list, start_ts, end_ts, window_ts, overlap_ts,
                                                use_imagery, eeg_2d_map, map_2d_shape):
    """
    Only use Left and Right fist data from MMIDB dataset

    :param base_route:
    :param subject_id_list:
    :param start_ts:
    :param end_ts:
    :param window_ts:
    :param overlap_ts:
    :param use_imagery:
    :param eeg_2d_map:
    :param map_2d_shape:
    :return:
    """
    mmidb_full_epoch_data, mmidb_full_label, mmidb_full_epoch_ts, mmidb_full_subjects = [], [], [], []
    if use_imagery:
        task_list = [4, 8, 12]
    else:
        task_list = [3, 7, 11]
    for subject_id in subject_id_list:
        subject_str = 'S%03d' % subject_id
        for task in task_list:
            task_str = subject_str + 'R%02d' % (task)
            saved_data_list = pickle.load(open(base_route + subject_str + '/' + task_str + '.p', 'rb'))
            label_list = pickle.load(open(base_route + subject_str + '/' + task_str + '_label.p', 'rb'))
            saved_2d_data_list = transform_slice_raw_data_2_2d(saved_data_list, eeg_2d_map, map_2d_shape)
            epoch_data_list, epoch_label_list, epoch_ts_list = epoch_2d_data_w_label(
                saved_2d_data_list, label_list, start_ts, end_ts, window_ts, overlap_ts)

            epoch_label_list = epoch_label_list - 2
            epoch_data_list = epoch_data_list[epoch_label_list >= 0, :, :, :]
            epoch_ts_list = epoch_ts_list[epoch_label_list >= 0, :]
            epoch_label_list = epoch_label_list[epoch_label_list >= 0]

            mmidb_full_epoch_data.append(epoch_data_list)
            mmidb_full_label.extend(epoch_label_list.tolist())
            mmidb_full_epoch_ts.append(epoch_ts_list)
            mmidb_full_subjects.extend([subject_id for _ in epoch_label_list])
    mmidb_full_epoch_data = np.concatenate(mmidb_full_epoch_data, axis=0)
    mmidb_full_label = np.array(mmidb_full_label)
    mmidb_full_epoch_ts = np.concatenate(mmidb_full_epoch_ts, axis=0)
    mmidb_full_subjects = np.array(mmidb_full_subjects)
    return mmidb_full_epoch_data, mmidb_full_label, mmidb_full_epoch_ts, mmidb_full_subjects


def generate_left_feet_classes_dataset_from_saved_data(base_route, subject_id_list, start_ts, end_ts, window_ts, overlap_ts,
                                                use_imagery, eeg_2d_map, map_2d_shape):
    """
    Only use 4 classes from MMIDB dataset

    :param base_route:
    :param subject_id_list:
    :param start_ts:
    :param end_ts:
    :param window_ts:
    :param overlap_ts:
    :param use_imagery:
    :param eeg_2d_map:
    :param map_2d_shape:
    :return:
    """
    mmidb_full_epoch_data, mmidb_full_label, mmidb_full_epoch_ts, mmidb_full_subjects = [], [], [], []
    if use_imagery:
        group_1_task_list = [4, 8, 12]
        group_2_task_list = [6, 10, 14]
    else:
        group_1_task_list = [3, 7, 11]
        group_2_task_list = [5, 9, 13]
    task_list = group_1_task_list + group_2_task_list
    for subject_id in subject_id_list:
        subject_str = 'S%03d' % subject_id
        for task in task_list:
            task_str = subject_str + 'R%02d' % (task)
            saved_data_list = pickle.load(open(base_route + subject_str + '/' + task_str + '.p', 'rb'))
            label_list = pickle.load(open(base_route + subject_str + '/' + task_str + '_label.p', 'rb'))
            saved_2d_data_list = transform_slice_raw_data_2_2d(saved_data_list, eeg_2d_map, map_2d_shape)
            epoch_data_list, epoch_label_list, epoch_ts_list = epoch_2d_data_w_label(
                saved_2d_data_list, label_list, start_ts, end_ts, window_ts, overlap_ts)  

            epoch_label_list = epoch_label_list - 2
            epoch_data_list = epoch_data_list[epoch_label_list >= 0, :, :, :]
            epoch_ts_list = epoch_ts_list[epoch_label_list >= 0, :]
            epoch_label_list = epoch_label_list[epoch_label_list >= 0]
            if task in group_2_task_list:
                epoch_label_list = epoch_label_list + 2
                epoch_data_list = epoch_data_list[epoch_label_list >= 2, :, :, :]
                epoch_ts_list = epoch_ts_list[epoch_label_list >= 2, :]
                epoch_label_list = epoch_label_list[epoch_label_list >= 2]
                # epoch_label_list = epoch_label_list - 1
            mmidb_full_epoch_data.append(epoch_data_list)
            mmidb_full_label.extend(epoch_label_list.tolist())
            mmidb_full_epoch_ts.append(epoch_ts_list)
            mmidb_full_subjects.extend([subject_id for _ in epoch_label_list])
    mmidb_full_epoch_data = np.concatenate(mmidb_full_epoch_data, axis=0)
    mmidb_full_label = np.array(mmidb_full_label)
    mmidb_full_epoch_ts = np.concatenate(mmidb_full_epoch_ts, axis=0)
    mmidb_full_subjects = np.array(mmidb_full_subjects)

    mmidb_full_epoch_data = mmidb_full_epoch_data[(mmidb_full_label==0) | (mmidb_full_label==3)]
    mmidb_full_epoch_ts = mmidb_full_epoch_ts[(mmidb_full_label==0) | (mmidb_full_label==3)]
    mmidb_full_subjects = mmidb_full_subjects[(mmidb_full_label==0) | (mmidb_full_label==3)]
    mmidb_full_label = mmidb_full_label[(mmidb_full_label==0) | (mmidb_full_label==3)]

    mmidb_full_label[mmidb_full_label==3] = 1


    return mmidb_full_epoch_data, mmidb_full_label, mmidb_full_epoch_ts, mmidb_full_subjects

def generate_right_feet_classes_dataset_from_saved_data(base_route, subject_id_list, start_ts, end_ts, window_ts, overlap_ts,
                                                use_imagery, eeg_2d_map, map_2d_shape):
    """
    Only use 4 classes from MMIDB dataset

    :param base_route:
    :param subject_id_list:
    :param start_ts:
    :param end_ts:
    :param window_ts:
    :param overlap_ts:
    :param use_imagery:
    :param eeg_2d_map:
    :param map_2d_shape:
    :return:
    """
    mmidb_full_epoch_data, mmidb_full_label, mmidb_full_epoch_ts, mmidb_full_subjects = [], [], [], []
    if use_imagery:
        group_1_task_list = [4, 8, 12]
        group_2_task_list = [6, 10, 14]
    else:
        group_1_task_list = [3, 7, 11]
        group_2_task_list = [5, 9, 13]
    task_list = group_1_task_list + group_2_task_list
    for subject_id in subject_id_list:
        subject_str = 'S%03d' % subject_id
        for task in task_list:
            task_str = subject_str + 'R%02d' % (task)
            saved_data_list = pickle.load(open(base_route + subject_str + '/' + task_str + '.p', 'rb'))
            label_list = pickle.load(open(base_route + subject_str + '/' + task_str + '_label.p', 'rb'))
            saved_2d_data_list = transform_slice_raw_data_2_2d(saved_data_list, eeg_2d_map, map_2d_shape)
            epoch_data_list, epoch_label_list, epoch_ts_list = epoch_2d_data_w_label(
                saved_2d_data_list, label_list, start_ts, end_ts, window_ts, overlap_ts)  

            epoch_label_list = epoch_label_list - 2
            epoch_data_list = epoch_data_list[epoch_label_list >= 0, :, :, :]
            epoch_ts_list = epoch_ts_list[epoch_label_list >= 0, :]
            epoch_label_list = epoch_label_list[epoch_label_list >= 0]
            if task in group_2_task_list:
                epoch_label_list = epoch_label_list + 2
                epoch_data_list = epoch_data_list[epoch_label_list >= 2, :, :, :]
                epoch_ts_list = epoch_ts_list[epoch_label_list >= 2, :]
                epoch_label_list = epoch_label_list[epoch_label_list >= 2]
                # epoch_label_list = epoch_label_list - 1
            mmidb_full_epoch_data.append(epoch_data_list)
            mmidb_full_label.extend(epoch_label_list.tolist())
            mmidb_full_epoch_ts.append(epoch_ts_list)
            mmidb_full_subjects.extend([subject_id for _ in epoch_label_list])
    mmidb_full_epoch_data = np.concatenate(mmidb_full_epoch_data, axis=0)
    mmidb_full_label = np.array(mmidb_full_label)
    mmidb_full_epoch_ts = np.concatenate(mmidb_full_epoch_ts, axis=0)
    mmidb_full_subjects = np.array(mmidb_full_subjects)

    mmidb_full_epoch_data = mmidb_full_epoch_data[(mmidb_full_label==1) | (mmidb_full_label==3)]
    mmidb_full_epoch_ts = mmidb_full_epoch_ts[(mmidb_full_label==1) | (mmidb_full_label==3)]
    mmidb_full_subjects = mmidb_full_subjects[(mmidb_full_label==1) | (mmidb_full_label==3)]
    mmidb_full_label = mmidb_full_label[(mmidb_full_label==1) | (mmidb_full_label==3)]

    mmidb_full_label[mmidb_full_label==1] = 0
    mmidb_full_label[mmidb_full_label==3] = 1


    return mmidb_full_epoch_data, mmidb_full_label, mmidb_full_epoch_ts, mmidb_full_subjects


def generate_eegmmidb_dataset_from_raw_edf(base_route, subject_id_list, start_ts, end_ts, window_ts, overlap_ts,
                                           use_imagery, use_no_movement, use_baseline_1, use_baseline_2,
                                           channel_mean, channel_std, eeg_2d_map, map_2d_shape):
    # type: (str, List[int], int, int, int, int, bool, bool, bool, bool, np.ndarray, np.ndarray, List[Tuple[int, int]], Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    """
    Generate Complete EEGMMIDB Epoch dataset

    :param base_route: base route to EEGMMIDB data
    :param subject_id_list: list of subjects
    :param start_ts: start timestep for each slice for epoch
    :param end_ts: maximum end timestep for each slice for epoch
    :param window_ts: window timestep for each epoch
    :param overlap_ts: overlap timestep between two window
    :param use_imagery: if true use imagery data instead of movement data
    :param use_no_movement: if true use no movement as class 0 in movement data
    :param use_baseline_1: if true use baseline task 1 also for non-movement class
    :param use_baseline_2: if true use baseline task 2 also for non-movement class
    :param channel_mean: mean for each channel
    :param channel_std: std for each channel
    :param eeg_2d_map: map for 2D transformation
    :param map_2d_shape: shape of the 2D map
    :return: mmidb_full_epoch_data, mmidb_full_label, mmidb_full_epoch_ts, mmidb_full_subjects
    """
    mmidb_full_epoch_data, mmidb_full_label, mmidb_full_epoch_ts, mmidb_full_subjects = [], [], [], []
    if use_imagery:
        group_1_task_list = [4, 8, 12]
        group_2_task_list = [6, 10, 14]
    else:
        group_1_task_list = [3, 7, 11]
        group_2_task_list = [5, 9, 13]
    group_0_task_list = []
    if use_baseline_1:
        group_0_task_list.append(1)
    if use_baseline_2:
        group_0_task_list.append(2)
    for subject_id in subject_id_list:
        for task in group_0_task_list:
            raw_data, events = read_raw_data_n_events(base_route, subject_id, task)
            raw_data_list, label_list = slice_raw_data_between_events(raw_data, events)
            norm_raw_data_list = normalize_slice_raw_data(raw_data_list, channel_mean, channel_std)
            raw_2d_data_list = transform_slice_raw_data_2_2d(norm_raw_data_list, eeg_2d_map, map_2d_shape)
            epoch_data_list, epoch_label_list, epoch_ts_list = epoch_2d_data_w_label(
                raw_2d_data_list, label_list, start_ts, raw_data.shape[1], window_ts, overlap_ts)
            epoch_label_list = epoch_label_list - 1
            mmidb_full_epoch_data.append(epoch_data_list)
            mmidb_full_label.extend(epoch_label_list.tolist())
            mmidb_full_epoch_ts.append(epoch_ts_list)
            mmidb_full_subjects.extend([subject_id for _ in epoch_label_list])
        for task in group_1_task_list:
            raw_data, events = read_raw_data_n_events(base_route, subject_id, task)
            raw_data_list, label_list = slice_raw_data_between_events(raw_data, events)
            norm_raw_data_list = normalize_slice_raw_data(raw_data_list, channel_mean, channel_std)
            raw_2d_data_list = transform_slice_raw_data_2_2d(norm_raw_data_list, eeg_2d_map, map_2d_shape)
            epoch_data_list, epoch_label_list, epoch_ts_list = epoch_2d_data_w_label(
                raw_2d_data_list, label_list, start_ts, end_ts, window_ts, overlap_ts)
            epoch_label_list = epoch_label_list - 1
            if not use_no_movement:
                epoch_data_list = epoch_data_list[epoch_label_list > 0, :, :, :]
                epoch_ts_list = epoch_ts_list[epoch_label_list > 0, :]
                epoch_label_list = epoch_label_list[epoch_label_list > 0]
            mmidb_full_epoch_data.append(epoch_data_list)
            mmidb_full_label.extend(epoch_label_list.tolist())
            mmidb_full_epoch_ts.append(epoch_ts_list)
            mmidb_full_subjects.extend([subject_id for _ in epoch_label_list])
        for task in group_2_task_list:
            raw_data, events = read_raw_data_n_events(base_route, subject_id, task)
            raw_data_list, label_list = slice_raw_data_between_events(raw_data, events)
            norm_raw_data_list = normalize_slice_raw_data(raw_data_list, channel_mean, channel_std)
            raw_2d_data_list = transform_slice_raw_data_2_2d(norm_raw_data_list, eeg_2d_map, map_2d_shape)
            epoch_data_list, epoch_label_list, epoch_ts_list = epoch_2d_data_w_label(
                raw_2d_data_list, label_list, start_ts, end_ts, window_ts, overlap_ts)
            epoch_label_list = epoch_label_list - 1
            epoch_label_list[epoch_label_list == 1] = 3
            epoch_label_list[epoch_label_list == 2] = 4
            if not use_no_movement:
                epoch_data_list = epoch_data_list[epoch_label_list > 0, :, :, :]
                epoch_ts_list = epoch_ts_list[epoch_label_list > 0, :]
                epoch_label_list = epoch_label_list[epoch_label_list > 0]
            mmidb_full_epoch_data.append(epoch_data_list)
            mmidb_full_label.extend(epoch_label_list.tolist())
            mmidb_full_epoch_ts.append(epoch_ts_list)
            mmidb_full_subjects.extend([subject_id for _ in epoch_label_list])
    mmidb_full_epoch_data = np.concatenate(mmidb_full_epoch_data, axis=0)
    mmidb_full_label = np.array(mmidb_full_label)
    mmidb_full_epoch_ts = np.concatenate(mmidb_full_epoch_ts, axis=0)
    mmidb_full_subjects = np.array(mmidb_full_subjects)
    return mmidb_full_epoch_data, mmidb_full_label, mmidb_full_epoch_ts, mmidb_full_subjects


def samples_per_class(labels):
    """
    Compute number of samples per class
    :param labels: numpy array of labels
    :return: num_per_class
    """
    num_per_class = [0 for _ in range(5)]
    for la in labels:
        num_per_class[la] += 1
    print("Number of samples per class: ", num_per_class)
    return num_per_class


def train_validate_split_subjects(mmidb_dataset, validation_subject_id_list):
    """
    Split data by training and validation using leave-k-out strategy
    :param mmidb_dataset: mmidb pytorch dataset
    :param validation_subject_id_list: list of subject id for validation
    :return: mmidb_train_item_indices, mmidb_val_item_indices
    """
    # print(mmidb_dataset.label.shape)
    _ = samples_per_class(mmidb_dataset.label)
    mmidb_train_item_indices = [
        item for item in range(len(mmidb_dataset)) if
        not (mmidb_dataset.epoch_subjects[item] in validation_subject_id_list)
    ]
    mmidb_val_item_indices = [
        item for item in range(len(mmidb_dataset)) if mmidb_dataset.epoch_subjects[item] in validation_subject_id_list
    ]
    return mmidb_train_item_indices, mmidb_val_item_indices



def sample_single_class_indices(mmidb_dataset, item_indices, class_label, sample_ratio):
    """
    Sample a certain class to a ratio in list of indices

    :param mmidb_dataset: mmidb pytorch dataset
    :param item_indices: list of item indices
    :param class_label: label of the class
    :param sample_ratio: sample ratio
    :return: sample_item_indices
    """
    sample_item_indices = copy.deepcopy(item_indices)
    for idx in sample_item_indices:
        if mmidb_dataset.label[idx] == class_label:
            tmp_sample = random()
            # If tmp_sample larger than sample ratio than drop it
            if tmp_sample > sample_ratio:
                sample_item_indices.remove(idx)
    return sample_item_indices

if __name__ == "__main__":
    route = '../data/'
    save_route = 'eegmmidb_slice_norm/'
    data_mean, data_std, data_max, data_min = compute_mean_std_for_all_data(route, 
            [i + 1 for i in range(109) if not (i + 1 in [88, 89, 92, 100, 104, 106])])
    np.save("mean_channel.npy", data_mean)
    np.save("std_channel.npy", data_std)

    generate_and_save_all_slice_data(route, save_route,
                                     [i + 1 for i in range(109) if not (i + 1 in [88, 89, 92, 100, 104, 106])],
                                     data_mean, data_std)


