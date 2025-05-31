import copy
import logging
import numpy as np
from tqdm import tqdm
import os
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from models.base import BaseLearner
from utils.inc_net import IncrementalNet, CreateNet, CSSRCriterion
from utils.toolkit import target2onehot, tensor2numpy
# import scipy.misc

EPSILON = 1e-8
init_milestones = [60, 120, 170]
milestones = [60, 90]
num_workers = 4
T = 2


class create(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = CreateNet(args, False)
        self.cssr_crt = CSSRCriterion()
        self.uce_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.args = args
        self.dataset = args["dataset"]
        self.batch_size = args["batch_size"]
        self.init_epoch = args["init_epoch"]
        self.epochs = args["epochs"]
        self.init_lr = args["init_lr"]
        self.init_lr_decay = args["init_lr_decay"]
        self.init_weight_decay = args["init_weight_decay"]
        self.lrate = args["lrate"]
        self.lrate_decay = args["lrate_decay"]
        self.weight_decay = args["weight_decay"] if args["weight_decay"] is not None else 0.0005
        self.ft_epochs = args["ft_epochs"]
        self.ft_lrate = args["ft_lrate"]
        self.init_cls = self.args["init_cls"]
        self.increment = self.args["increment"]

        self.cont_beta = self.args["cont_beta"]

        self.step_type = self.args["step_type"]  # CosineAnnealingLR

        try:
            print(self.args["pretrain_path"])
        except:
            self.args["pretrain_path"] = "None"

        try:
            self.mile = self.args["mile"]
        except:
            self.mile = False

        try:
            self.mile2 = self.args["mile2"]
        except:
            self.mile2 = False

        self.cont_weight = self.args["cont_weight"]
        self.unc_weight = 1.0

        if self.args["freeze_total_bkb"]:
            logging.info("Rebuttal: freeze_total_bkb!")
        if self.args["freezeold"]:
            logging.info("Rebuttal: freeze old AE!")
        self.setup_gpu(args, os)

    def after_task(self):
        if len(self._multiple_gpus) > 1:
            self._old_network = self._network.module.copy().freeze()
        else:
            self._old_network = self._network.copy().freeze()
        self._known_classes = self._total_classes
        logging.info("Exemplar size: {}".format(self.exemplar_size))

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        self.task_size = self._total_classes - self._known_classes
        if len(self._multiple_gpus) == 1 or self._cur_task == 0:
            self._network.update_fc(self._total_classes)
            self._network.unfreeze_bkb()
            self._network.unfrz_fc(self._total_classes)
        else:
            self._network.module.update_fc(self._total_classes)
            self._network.module.unfreeze_bkb()
            self._network.module.unfrz_fc(self._total_classes)

        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )
        if self._cur_task >= 1:
            self.lamda = self.args["kdlamda"]
            logging.info("Lambda: {:.3f}".format(self.lamda))

        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
            appendent=self._get_memory(),
        )
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
        )
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
        )

        all_observed_train_dset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="train", mode="train"
        )
        self.all_observed_train_loader = DataLoader(
            all_observed_train_dset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers
        )

        if len(self._multiple_gpus) > 1 and self._cur_task == 0:
            self._network = nn.DataParallel(self._network, device_ids=list(range(torch.cuda.device_count())))

        self._train(self.train_loader, self.test_loader, data_manager)
        self.build_rehearsal_memory(data_manager, self.samples_per_class)


    def _train(self, train_loader, test_loader, data_manager):
        self._network.cuda()
        if self._old_network is not None:
            self._old_network.cuda()

        if self._cur_task == 0:
            optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                momentum=0.9,
                lr=self.init_lr,
                weight_decay=self.init_weight_decay,
            )
            if self.step_type == 'MultiStepLR':
                scheduler = optim.lr_scheduler.MultiStepLR(
                    optimizer=optimizer, milestones=init_milestones, gamma=self.init_lr_decay
                )
            elif self.step_type == 'CosineAnnealingLR':
                scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer=optimizer, T_max=self.init_epoch)

            # load existing model
            pretrain_path = self.args["pretrain_path"]
            if os.path.exists(pretrain_path):
                logging.info("have pretrained model")
                if len(self._multiple_gpus) == 1:
                    self._network.load_state_dict(torch.load(pretrain_path, map_location='cuda:0'))
                elif len(self._multiple_gpus) > 1:
                    checkpoint = torch.load(pretrain_path)
                    try:
                        self._network.load_state_dict({'module.'+k: v for k, v in checkpoint.items()})
                    except:
                        self._network.load_state_dict({k: v for k, v in checkpoint.items()})
                train_tag = False
                save_tag = False
            else:
                train_tag = True
                save_tag = True
            self._init_train(train_loader, test_loader, optimizer, scheduler, train_tag=train_tag, save_tag=save_tag)
        else:
            if self.mile and self._cur_task == self.args["milephase"]:
                self.lrate = self.lrate * 0.1
                logging.info("change lr to {} in {} phase".format(self.lrate, self.args["milephase"]))
                self.mile = False
            if self.mile2 and self._cur_task == self.args["milephase2"]:
                self.lrate = self.lrate * 0.1
                logging.info("change lr to {} in {} phase".format(self.lrate, self.args["milephase2"]))
                self.mile2 = False

            optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                lr=self.lrate,
                momentum=0.9,
                weight_decay=self.weight_decay,
            )
            if self.step_type == 'MultiStepLR':
                scheduler = optim.lr_scheduler.MultiStepLR(
                    optimizer=optimizer, milestones=milestones, gamma=self.lrate_decay
                )
            elif self.step_type == 'CosineAnnealingLR':
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.epochs, eta_min=1e-8)
            self._update_representation(train_loader, test_loader, optimizer, scheduler, self.epochs, ftmode=False)

            # start balanced training stage
            logging.info(
                "Finetune the network (classifier part) with the undersampled dataset!"
            )
            if len(self._multiple_gpus) == 1:
                self._network.unfrz_fc(self._total_classes)
            else:
                self._network.module.unfrz_fc(self._total_classes)


            if self._fixed_memory:
                finetune_samples_per_class = self._memory_per_class
                self._construct_exemplar_unified(data_manager, finetune_samples_per_class)
            else:
                finetune_samples_per_class = self._memory_size // self._known_classes
                self._reduce_exemplar(data_manager, finetune_samples_per_class)
                self._construct_exemplar(data_manager, finetune_samples_per_class)

            finetune_train_dataset = data_manager.get_dataset(
                [], source="train", mode="train", appendent=self._get_memory()
            )
            finetune_train_loader = DataLoader(
                finetune_train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory = True
            )
            logging.info(
                "The size of finetune dataset: {}".format(len(finetune_train_dataset))
            )
            if len(self._multiple_gpus) == 1:
                self._network.freeze_bkb('total')
                network_params = [
                    {"params": self._network.convnet.parameters(), "lr": 0, "weight_decay": 0},
                    {"params": self._network.fc.parameters(), "lr": self.ft_lrate,
                     "weight_decay": self.weight_decay}
                ]
            else:
                self._network.module.freeze_bkb('total')
                network_params = [
                    {"params": self._network.module.convnet.parameters(), "lr": 0, "weight_decay": 0},
                    {"params": self._network.module.fc.parameters(), "lr": self.ft_lrate,
                     "weight_decay": self.weight_decay}
                ]

            optimizer = optim.SGD(
                network_params, lr=self.ft_lrate, momentum=0.9, weight_decay=self.weight_decay
            )
            if self.step_type == 'MultiStepLR':
                scheduler = optim.lr_scheduler.MultiStepLR(
                    optimizer=optimizer, milestones=milestones, gamma=self.lrate_decay
                )
            elif self.step_type == 'CosineAnnealingLR':
                scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer=optimizer, T_max=self.ft_epochs
                )

            self._update_representation(finetune_train_loader, test_loader, optimizer, scheduler, self.ft_epochs, ftmode=True)

            if self._fixed_memory and self.task_size != 0:
                self._data_memory = self._data_memory[
                                    : -self._memory_per_class * self.task_size
                                    ]
                self._targets_memory = self._targets_memory[
                                       : -self._memory_per_class * self.task_size
                                       ]
                assert (
                        len(
                            np.setdiff1d(
                                self._targets_memory, np.arange(0, self._known_classes)
                            )
                        )
                        == 0
                ), "Exemplar error!"
            # balance training stage end


    def _init_train(self, train_loader, test_loader, optimizer, scheduler, train_tag=True, save_tag=False):
        prog_bar = tqdm(range(self.init_epoch))
        if not train_tag:
            info = "load existing model"
        else:
            for _, epoch in enumerate(prog_bar):
                self._network.train()
                losses, losses_clf, losses_cons = 0.0, 0.0, 0.0
                correct, total = 0, 0
                for i, (_, inputs, targets) in tqdm(enumerate(train_loader), total=len(train_loader)):
                    inputs, targets = inputs.cuda(), targets.cuda()  # go into inc_net!
                    outputs = self._network(inputs)
                    logits = outputs["logits"]
                    fms = outputs["fm"]
                    loss_clf = self.cssr_crt(logits, targets)

                    s, w = map_to_weight(outputs["error"], self._device, alpha=self.cont_beta)
                    if self.unc_weight == 0:
                        loss_cons = Contrastive_Loss(fms, targets, self._device, w,
                                                     use_w=False) * self.cont_weight
                    else:
                        loss_cons = Contrastive_Loss(fms, targets, self._device, w) * self.cont_weight

                    loss = loss_clf + loss_cons

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    losses += loss.item()
                    losses_clf += loss_clf.item()
                    losses_cons += loss_cons.item()

                    _, preds = torch.max(logits, dim=1)
                    correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                    total += len(targets)

                scheduler.step()
                train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

                if epoch % 5 == 0:
                    test_acc, cnn_accy = self._compute_accuracy(self._network, test_loader)
                    info = "Task {}, Epoch {}/{} => Losses_clf {:.3f}, losses_cons {:.5f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                        self._cur_task,
                        epoch + 1,
                        self.init_epoch,
                        losses_clf / len(train_loader),
                        losses_cons / len(train_loader),
                        train_acc,
                        test_acc,
                    )
                    logging.info(cnn_accy["grouped"])
                else:
                    info = "Task {}, Epoch {}/{} => Loss {:.3f}, Loss_clf {:.3f}, losses_cons {:.5f}, Train_accy {:.2f}".format(
                        self._cur_task,
                        epoch + 1,
                        self.init_epoch,
                        losses / len(train_loader),
                        losses_clf / len(train_loader),
                        losses_cons / len(train_loader),
                        train_acc,
                    )
                logging.info(info)
                prog_bar.set_description(info)

        logging.info(info)

        # save
        if save_tag:
            torch.save(self._network.state_dict(), self.args["pretrain_path"])
            print("Save base model to {}".format(self.args["pretrain_path"]))

        # calculate prototypes of fm
        self._network.eval()
        with torch.no_grad():
            self.class_fms_sums = torch.zeros(self._total_classes, self.args['ae_latent']).cuda()
            self.class_counts = torch.zeros(self._total_classes).cuda()
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.cuda(), targets.cuda()
                outputs = self._network(inputs)
                fms = outputs["fm"]  # torch.Size([128, 50, 32])
                self.update_fms_prototypes(fms, targets)
            self.fms_prototypes = self.compute_prototypes()


    def update_fms_prototypes(self, data, labels):
        for i in range(self._total_classes):
            class_mask = (labels == i)
            class_data = data[class_mask]
            class_data = class_data[:, i, :]
            # normalization
            class_data = (class_data.T / (class_data.T.norm(dim=0) + EPSILON)).T
            class_count = class_data.size(0)
            if class_count > 0:
                self.class_counts[i] += class_count
                self.class_fms_sums[i] += torch.sum(class_data, dim=0)

    def compute_prototypes(self):
        fms_prototypes = self.class_fms_sums / self.class_counts.unsqueeze(1).clamp(min=1) # torch.Size([50, 32])
        return fms_prototypes.cuda()

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler, epochs, ftmode=False):
        prog_bar = tqdm(range(epochs))
        for _, epoch in enumerate(prog_bar):
            self._network.train()

            # Rebuttal exps
            if self.args["freeze_total_bkb"]:
                self._network.freeze_bkb('total')
            if self.args["freezeold"]:
                self._network.freeze_oldAEs()

            losses, losses_clf, losses_kd, losses_cr = 0.0, 0.0, 0.0, 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                self.class_fms_sums = torch.zeros(self._total_classes, self.args['ae_latent']).cuda()
                self.class_counts = torch.zeros(self._total_classes).cuda()
                loss_cr = torch.tensor(0.0)

                inputs, targets = inputs.cuda(), targets.cuda()
                outputs = self._network(inputs)
                fms = outputs["fm"]
                logits = outputs["logits"]  # 经过了softmax的，已经是概率了
                loss_clf = self.cssr_crt(logits, targets)  # TODO notice

                with torch.no_grad():
                    old_output = self._old_network(inputs)  # error -> logits
                    old_logits = old_output["error"]
                if len(_) > 1:
                    loss_kd = _KD_loss(
                        outputs["error"][:, : self._known_classes],
                        old_logits,
                        T,
                    ) * self.lamda
                else:
                    loss_kd = _KD_loss(
                        outputs["error"].unsqueeze(0)[:, : self._known_classes],
                        old_logits.unsqueeze(0),
                        T,
                    ) * self.lamda


                if self.args["cr_loss"] and len(_) > 1:  # 旧类的保持
                    self.update_fms_prototypes(fms, targets)
                    self.new_fms_prototypes = self.compute_prototypes()

                    w = None
                    s, w = map_to_weight(outputs["error"], self._device, alpha=self.cont_beta)  # (128,) (128,)
                    if self.unc_weight == 0:
                        loss_cons = Contrastive_Loss(fms, targets, self._device, w,
                                                     use_w=False) * self.cont_weight
                    else:
                        loss_cons = Contrastive_Loss(fms, targets, self._device, w) * self.cont_weight
                    loss_cr = loss_cons

                loss = loss_clf + loss_kd + loss_cr

                optimizer.zero_grad()
                loss.backward()

                optimizer.step()
                losses += loss.item()
                losses_clf += loss_clf.item()
                losses_kd += loss_kd.item()
                try:
                    losses_cr += (
                        loss_cr.item() if self.args["cr_loss"] else loss_cr
                    )
                except:
                    print(loss_cr)

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            if epoch % 5 == 0:
                test_acc, cnn_accy = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Loss_ce {:.3f}, Loss_kd {:.3f}, " \
                       "Loss_cr {:.5f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                            self._cur_task,
                            epoch + 1,
                            epochs,
                            losses / len(train_loader),
                            losses_clf / len(train_loader),
                            losses_kd / len(train_loader),
                            losses_cr / len(train_loader),
                            train_acc,
                            test_acc,
                        )
                logging.info(cnn_accy["grouped"])
            else:
                test_acc, cnn_accy = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Loss_ce {:.3f}, Loss_kd {:.3f}, " \
                       "Loss_cr {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                            self._cur_task,
                            epoch + 1,
                            epochs,
                            losses / len(train_loader),
                            losses_clf / len(train_loader),
                            losses_kd / len(train_loader),
                            losses_cr / len(train_loader),
                            train_acc,
                            test_acc
                        )
                logging.info(cnn_accy["grouped"])
            logging.info(info)
            prog_bar.set_description(info)
        logging.info(info)

        self._network.eval()
        with torch.no_grad():
            self.class_fms_sums = torch.zeros(self._total_classes, self.args['ae_latent']).cuda()
            self.class_counts = torch.zeros(self._total_classes).cuda()
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.cuda(), targets.cuda()
                outputs = self._network(inputs)
                fms = outputs["fm"]  # torch.Size([128, 50, 32])
                if len(_) > 1:
                    self.update_fms_prototypes(fms, targets)

            self.fms_prototypes = self.compute_prototypes()


def _KD_loss(pred, soft, T):
    pred = torch.log_softmax(pred / T, dim=1)
    soft = torch.softmax(soft / T, dim=1)
    return -1 * torch.mul(soft, pred).sum() * T * T / pred.shape[0]


def map_to_weight(error, device, alpha=-10.0):
    values, _ = torch.sort(error, dim=1, descending=True)
    first_max = values[:, 0]
    second_max = values[:, 1]
    first_min = values[:, -1]
    s = torch.abs(second_max - first_max) / torch.abs(first_max - first_min + EPSILON)
    s = s.detach().cpu().numpy()
    w = 1 + np.exp(-1.0 * alpha * s)  # w: (128, 1)
    w = torch.from_numpy(w).cuda()
    return s, w


def Contrastive_Loss(features, labels, device, sample_w=None, use_w=True, temperature=0.1, base_temperature=0.07):
    '''
    :param sample_w: weight of samples
    :param mode: h: horizontal, v: vertical
    '''
    # features: torch.Size([128, 50, 32])
    batch_size = features.shape[0]
    num_AEs = features.shape[1]
    final_loss = 0.0
    h_loss, v_loss, p_loss = 0.0, 0.0, 0.0
    if sample_w is None:
        weights_cls = torch.ones(num_AEs, 1).cuda()
    else:
        weights_cls = torch.zeros(num_AEs, 1).cuda()
        for i in range(num_AEs):
            mask = (labels == i)  # Mask of samples belonging to the i-th class
            if mask.sum().item() > 0:
                weights_cls[i] = torch.max(sample_w[mask])
    if use_w == False:
        sample_w = torch.ones(batch_size, 1).cuda()
        sample_w = sample_w.squeeze(1)
        weights_cls = torch.ones(num_AEs, 1).cuda()

    contrast_count = 1
    for i in range(num_AEs):
        mask = (labels.unsqueeze(1) == labels.unsqueeze(0)) & (labels.unsqueeze(1) == i)
        contrast_feature = features[:,i,:]

        anchor_feature = contrast_feature
        anchor_count = contrast_count

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)  # (128, 1)
        logits = anchor_dot_contrast - logits_max.detach()  # torch.Size([128, 128])

        logits = torch.clamp(logits, min=-95.0)

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).cuda(),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        loss = - (temperature / base_temperature) * mean_log_prob_pos * sample_w
        h_loss += loss.view(anchor_count, batch_size).mean()
    h_loss = h_loss / num_AEs

    final_loss = h_loss
    return final_loss