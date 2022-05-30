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
    Split data by training and validation using leave-one-out strategy
    :param mmidb_dataset: mmidb pytorch dataset
    :param validation_subject_id_list: list of subject id for validation
    :return: mmidb_train_item_indices, mmidb_val_item_indices
    """
    _ = samples_per_class(mmidb_dataset.label)
    mmidb_train_item_indices = [
        item for item in range(len(mmidb_dataset)) if
        not (mmidb_dataset.epoch_subjects[item] in validation_subject_id_list)
    ]
    mmidb_val_item_indices = [
        item for item in range(len(mmidb_dataset)) if mmidb_dataset.epoch_subjects[item] in validation_subject_id_list
    ]
    return mmidb_train_item_indices, mmidb_val_item_indices
