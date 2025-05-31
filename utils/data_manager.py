import logging

import dill
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from utils.data import iCIFAR10, iCIFAR100, iCIFAR224, iImageNet100, iImageNet1000, i102flowers, CUB, UCF101, SUN
from tqdm import tqdm

class DataManager(object):
    def __init__(self, dataset_name, shuffle, seed, init_cls, increment, scenario='normal', scenario_config=None):
        self.dataset_name = dataset_name
        self.scenario = scenario

        if self.scenario == 'normal' or self.scenario == 'collective':
            self._setup_data(dataset_name, shuffle, seed)  # including getting everything about dataset and re-mapping the label
            assert init_cls <= len(self._class_order), "No enough classes."
            self._increments = [init_cls]
            while sum(self._increments) + increment < len(self._class_order):
                self._increments.append(increment)
            offset = len(self._class_order) - sum(self._increments)
            if offset > 0:
                self._increments.append(offset)
        elif self.scenario == 'challenge':  # challenge
            with open(f"../challenge/scenario_configs/{scenario_config}", "rb") as pkl_file:
                self.scenario_config = dill.load(pkl_file)
            self.scenario_table = self.scenario_config['scenario_table']
            self.n_samples_table = self.scenario_config['n_samples_table']
            self.first_occurrences = self.scenario_config['first_occurrences']
            self.indices_per_class = self.scenario_config['indices_per_class']
            self.n_classes = self.scenario_config['n_classes']
            self.n_e = self.scenario_config['n_e']
            self._increments, previous_classes, order_list = [], [], []
            self.train_lst, self.test_lst, self.newclasses_exp = [], [], []
            for exp_i in range(self.n_e):
                classes_in_this_experience = torch.where(self.scenario_table[:, exp_i] != 0)[0].numpy().tolist()
                newclasses_in_this_experience = list(set(classes_in_this_experience).difference(set(previous_classes)))
                order_list.extend(newclasses_in_this_experience)
                self.newclasses_exp.append(newclasses_in_this_experience)
                self.train_lst.append(classes_in_this_experience)
                self.test_lst.append(newclasses_in_this_experience)
                self._increments.append(len(newclasses_in_this_experience))
                previous_classes = list(set(previous_classes + classes_in_this_experience))
            self._setup_data(dataset_name, shuffle, seed, order=order_list)
            for i in range(self.n_e) :
                self.newclasses_exp[i] = [self.class_order_dict[class_val] for class_val in self.newclasses_exp[i]]

    @property
    def nb_tasks(self):
        if self.scenario == 'normal' or self.scenario == 'collective':
            return len(self._increments)
        elif self.scenario == 'challenge':
            return self.n_e

    def get_task_size(self, task):
        return self._increments[task]

    def get_accumulate_tasksize(self,task):
        return sum(self._increments[:task+1])
    
    def get_total_classnum(self):
        return len(self._class_order)

    def get_indices(self, exp_i, source):
        if source == "train":
            present_classes = self.train_lst[exp_i]
            order_indices = [self.class_order_dict[class_val] for class_val in present_classes]
            indices = np.array(order_indices)
            logging.info("present_classes: {}\ntrain exp_i: {}, indices: {}".format(present_classes, exp_i, indices))
        elif source == "test":
            present_classes = []
            for k in range(exp_i+1):
                present_classes.extend(self.test_lst[k])
            order_indices = [self.class_order_dict[class_val] for class_val in present_classes]
            indices = np.array(order_indices)
            logging.info("test exp_i: {}, indices: {}".format(exp_i, indices))
        elif source == "mm-memory":
            present_classes = self.train_lst[exp_i]
            order_indices = [self.class_order_dict[class_val] for class_val in present_classes]
            indices = np.array(order_indices)
            logging.info("mm-memory exp_i: {}, indices: {}".format(exp_i, indices))
        return indices

    def get_challenge_dataset(
        self, exp_i, indices, source, mode, appendent=None, ret_data=False, m_rate=None
    ):
        if source == "train":
            x, y = self._train_data, self._train_targets
        elif source == "test":
            x, y = self._test_data, self._test_targets
        elif source == "mm-memory":  #  New Accessed by base.py line 80
            x, y = appendent
        else:
            raise ValueError("Unknown data source {}.".format(source))

        if mode == "train":
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
        elif mode == "flip":
            trsf = transforms.Compose(
                [
                    *self._test_trsf,
                    transforms.RandomHorizontalFlip(p=1.0),
                    *self._common_trsf,
                ]
            )
        elif mode == "test":
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
        else:
            raise ValueError("Unknown mode {}.".format(mode))

        data, targets = [], []
        for idx in indices:
            if m_rate is None:
                class_data, class_targets = self._select(
                    x, y, low_range=idx, high_range=idx + 1
                )
            else:
                class_data, class_targets = self._select_rmm(
                    x, y, low_range=idx, high_range=idx + 1, m_rate=m_rate
                )
            # 根据n_samples_table选取样本
            if source == "train":
                idx_n = self.n_samples_table[self.order_class_dict[idx]][exp_i]
                class_data = class_data[torch.randperm(len(class_data))][:idx_n]
                data.append(class_data)
                targets.append(class_targets[:idx_n])
            elif source == "test":
                data.append(class_data)
                targets.append(class_targets)

        if appendent is not None and len(appendent) != 0 and source != "mm-memory":
            appendent_data, appendent_targets = appendent
            data.append(appendent_data)
            targets.append(appendent_targets)

        data, targets = np.concatenate(data), np.concatenate(targets)

        if ret_data:
            return data, targets, DummyDataset(data, targets, trsf, self.use_path)
        else:
            return DummyDataset(data, targets, trsf, self.use_path)

    def get_dataset(
        self, indices, source, mode, appendent=None, ret_data=False, m_rate=None
    ):
        if source == "train":
            x, y = self._train_data, self._train_targets
        elif source == "test":
            x, y = self._test_data, self._test_targets
        elif source == "mm-memory":  # New Accessed by base.py line 80
            x, y = appendent
        else:
            raise ValueError("Unknown data source {}.".format(source))

        if mode == "train":
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
        elif mode == "flip":
            trsf = transforms.Compose(
                [
                    *self._test_trsf,
                    transforms.RandomHorizontalFlip(p=1.0),
                    *self._common_trsf,
                ]
            )
        elif mode == "test":
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
        else:
            raise ValueError("Unknown mode {}.".format(mode))

        data, targets = [], []
        for idx in indices:
            if m_rate is None:
                class_data, class_targets = self._select(
                    x, y, low_range=idx, high_range=idx + 1
                )
            else:
                class_data, class_targets = self._select_rmm(
                    x, y, low_range=idx, high_range=idx + 1, m_rate=m_rate
                )
            data.append(class_data)
            targets.append(class_targets)

        if appendent is not None and len(appendent) != 0 and source != "mm-memory":
            appendent_data, appendent_targets = appendent
            data.append(appendent_data)
            targets.append(appendent_targets)

        data, targets = np.concatenate(data), np.concatenate(targets)

        if ret_data:
            return data, targets, DummyDataset(data, targets, trsf, self.use_path)
        else:
            return DummyDataset(data, targets, trsf, self.use_path)

    def get_finetune_dataset(self,known_classes,total_classes,source,mode,appendent,type="ratio"):
        if source == 'train':
            x, y = self._train_data, self._train_targets
        elif source == 'test':
            x, y = self._test_data, self._test_targets
        else:
            raise ValueError('Unknown data source {}.'.format(source))

        if mode == 'train':
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
        elif mode == 'test':
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
        else:
            raise ValueError('Unknown mode {}.'.format(mode))
        val_data = []
        val_targets = []

        old_num_tot = 0
        appendent_data, appendent_targets = appendent

        for idx in range(0, known_classes):
            append_data, append_targets = self._select(appendent_data, appendent_targets,
                                                       low_range=idx, high_range=idx+1)
            num=len(append_data)
            if num == 0:
                continue
            old_num_tot += num
            val_data.append(append_data)
            val_targets.append(append_targets)
        if type == "ratio":
            new_num_tot = int(old_num_tot*(total_classes-known_classes)/known_classes)
        elif type == "same":
            new_num_tot = old_num_tot
        else:
            assert 0, "not implemented yet"
        new_num_average = int(new_num_tot/(total_classes-known_classes))
        for idx in range(known_classes,total_classes):
            class_data, class_targets = self._select(x, y, low_range=idx, high_range=idx+1)
            val_indx = np.random.choice(len(class_data),new_num_average, replace=False)
            val_data.append(class_data[val_indx])
            val_targets.append(class_targets[val_indx])
        val_data=np.concatenate(val_data)
        val_targets = np.concatenate(val_targets)
        return DummyDataset(val_data, val_targets, trsf, self.use_path)

    def get_dataset_with_split(
        self, indices, source, mode, appendent=None, val_samples_per_class=0
    ):
        if source == "train":
            x, y = self._train_data, self._train_targets
        elif source == "test":
            x, y = self._test_data, self._test_targets
        else:
            raise ValueError("Unknown data source {}.".format(source))

        if mode == "train":
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
        elif mode == "test":
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
        else:
            raise ValueError("Unknown mode {}.".format(mode))

        train_data, train_targets = [], []
        val_data, val_targets = [], []
        for idx in indices:
            class_data, class_targets = self._select(
                x, y, low_range=idx, high_range=idx + 1
            )
            val_indx = np.random.choice(
                len(class_data), val_samples_per_class, replace=False
            )
            train_indx = list(set(np.arange(len(class_data))) - set(val_indx))
            val_data.append(class_data[val_indx])
            val_targets.append(class_targets[val_indx])
            train_data.append(class_data[train_indx])
            train_targets.append(class_targets[train_indx])

        if appendent is not None:
            appendent_data, appendent_targets = appendent
            for idx in range(0, int(np.max(appendent_targets)) + 1):
                append_data, append_targets = self._select(
                    appendent_data, appendent_targets, low_range=idx, high_range=idx + 1
                )
                val_indx = np.random.choice(
                    len(append_data), val_samples_per_class, replace=False
                )
                train_indx = list(set(np.arange(len(append_data))) - set(val_indx))
                val_data.append(append_data[val_indx])
                val_targets.append(append_targets[val_indx])
                train_data.append(append_data[train_indx])
                train_targets.append(append_targets[train_indx])

        train_data, train_targets = np.concatenate(train_data), np.concatenate(
            train_targets
        )
        val_data, val_targets = np.concatenate(val_data), np.concatenate(val_targets)

        # return DummyDataset(
        #     train_data, train_targets, trsf, self.use_path
        # ), DummyDataset(val_data, val_targets, trsf, self.use_path)
        return DummyDataset(
            train_data, train_targets, trsf, self.use_path
        ), DummyDataset(val_data, val_targets, trsf, self.use_path)

    def _setup_data(self, dataset_name, shuffle, seed, order=None):
        idata = _get_idata(dataset_name)
        idata.download_data()

        # Data
        self._train_data, self._train_targets = idata.train_data, idata.train_targets
        self._test_data, self._test_targets = idata.test_data, idata.test_targets
        self.use_path = idata.use_path

        # Transforms
        self._train_trsf = idata.train_trsf
        self._test_trsf = idata.test_trsf
        self._common_trsf = idata.common_trsf

        # Order
        if order == None:
            order = [i for i in range(len(np.unique(self._train_targets)))]
            if shuffle:
                np.random.seed(seed)
                order = np.random.permutation(len(order)).tolist()
            else:
                order = idata.class_order
            self._class_order = order
        else:
            self._class_order = order
        self.class_order_dict = {class_val: idx for idx, class_val in enumerate(self._class_order)}  # {4: 0, 6: 1, 9: 2, ...}
        self.order_class_dict = {idx: class_val for idx, class_val in enumerate(self._class_order)}  # {0: 4, 1: 6, 2: 9, ...}
        logging.info(self._class_order)

        # Map indices
        self._train_targets = _map_new_class_index(
            self._train_targets, self._class_order
        )
        self._test_targets = _map_new_class_index(self._test_targets, self._class_order)

    def _select(self, x, y, low_range, high_range):
        idxes = np.where(np.logical_and(y >= low_range, y < high_range) )[0]
        
        if isinstance(x,np.ndarray):
            x_return = x[idxes]
        else:
            x_return = []
            for id in idxes:
                x_return.append(x[id])
        return x_return, y[idxes]

    def _select_rmm(self, x, y, low_range, high_range, m_rate):
        assert m_rate is not None
        if m_rate != 0:
            idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
            selected_idxes = np.random.randint(
                0, len(idxes), size=int((1 - m_rate) * len(idxes))
            )
            new_idxes = idxes[selected_idxes]
            new_idxes = np.sort(new_idxes)
        else:
            new_idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        return x[new_idxes], y[new_idxes]

    def getlen(self, index):
        y = self._train_targets
        return np.sum(np.where(y == index))


class DummyDataset(Dataset):
    def __init__(self, images, labels, trsf, use_path=False):
        assert len(images) == len(labels), "Data size error!"
        self.images = images
        self.labels = labels
        self.trsf = trsf
        self.use_path = use_path

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.use_path:
            image = self.trsf(pil_loader(self.images[idx]))
        else:
            image = self.trsf(Image.fromarray(self.images[idx]))
        label = self.labels[idx]

        return idx, image, label


def _map_new_class_index(y, order):
    return np.array(list(map(lambda x: order.index(x), y)))


def _get_idata(dataset_name):
    name = dataset_name.lower()
    if name == "cifar10":
        return iCIFAR10()
    elif name == "cifar100":
        return iCIFAR100()
    elif name == "cifar224":
        return iCIFAR224()
    elif name == "imagenet1000":
        return iImageNet1000()
    elif name == "imagenet100":
        return iImageNet100()
    elif name == "102flowers":
        return i102flowers()
    elif name == "cub":
        return CUB()
    elif name == "ucf":
        return UCF101()
    elif name == "sun":
        return SUN()
    else:
        raise NotImplementedError("Unknown dataset {}.".format(dataset_name))


def pil_loader(path):
    """
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    """
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def accimage_loader(path):
    """
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    accimage is an accelerated Image loader and preprocessor leveraging Intel IPP.
    accimage is available on conda-forge.
    """
    import accimage

    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    """
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    """
    from torchvision import get_image_backend

    if get_image_backend() == "accimage":
        return accimage_loader(path)
    else:
        return pil_loader(path)
