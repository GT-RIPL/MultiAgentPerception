import os
import yaml
import time
import shutil
import torch
import random
import argparse
import numpy as np
import copy
import timeit
import statistics
import datetime
from torch.utils import data
from tqdm import tqdm
import cv2

from ptsemseg.process_img import generate_noise
from ptsemseg.models import get_model
from ptsemseg.loss import get_loss_function
from ptsemseg.loader import get_loader
from ptsemseg.utils import get_logger, init_weights
from ptsemseg.metrics import runningScore, averageMeter
from ptsemseg.augmentations import get_composed_augmentations
from ptsemseg.schedulers import get_scheduler
from ptsemseg.optimizers import get_optimizer
from ptsemseg.utils import convert_state_dict

from tensorboardX import SummaryWriter


class Trainer_LearnWhen2Com(object):
    def __init__(self, cfg, writer, logger, model, loss_fn, trainloader, valloader, optimizer, scheduler, device):

        self.cfg = cfg
        self.writer = writer
        self.logger = logger
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn

        self.n_classes = 11
        self.MO_flag = self.cfg['model']['multiple_output']

        self.running_metrics_val = runningScore(self.n_classes)
        self.device = device

        if 'commun_label' in self.cfg["data"]:
            self.if_commun_label = cfg["data"]['commun_label']
        else:
            self.if_commun_label = 'None'

    def train(self):
        start_iter = 0

        # resume the training
        if self.cfg["training"]["resume"] is not None:
            if os.path.isfile(self.cfg["training"]["resume"]):
                self.logger.info(
                    "Loading model and optimizer from checkpoint '{}'".format(cfg["training"]["resume"])
                )
                checkpoint = torch.load(self.cfg["training"]["resume"])
                self.model.load_state_dict(checkpoint["model_state"])
                self.optimizer.load_state_dict(checkpoint["optimizer_state"])
                self.scheduler.load_state_dict(checkpoint["scheduler_state"])
                start_iter = checkpoint["epoch"]
                self.logger.info(
                    "Loaded checkpoint '{}' (iter {})".format(
                        self.cfg["training"]["resume"], checkpoint["epoch"]
                    )
                )
            else:
                self.logger.info("No checkpoint found at '{}'".format(self.cfg["training"]["resume"]))


        val_loss_meter = averageMeter()
        time_meter = averageMeter()

        best_iou = -100.0
        i = start_iter
        flag = True

        # Training
        while i <= self.cfg["training"]["train_iters"] and flag:
            for data_list in self.trainloader:
                # iteration timer
                i += 1

                # load data from dataloader
                if self.if_commun_label != 'None':
                    images_list, labels_list, commun_label = data_list
                else:
                    images_list, labels_list = data_list

                # image and labels list 2 tensor
                labels = labels_list[0]
                images = torch.cat(tuple(images_list), dim=1)

                # timer started
                start_ts = time.time()

                self.scheduler.step()
                self.model.train()  # matters for batchnorm/dropout

                # from cpu to gpu
                images = images.to(self.device)
                labels = labels.to(self.device)
                if self.if_commun_label != 'None':
                    commun_label = commun_label.to(self.device)

                # clean the optimizer
                self.optimizer.zero_grad()

                # model inference
                outputs, log_action, action_argmax = self.model(images, training=True)

                # compute loss
                loss = self.loss_fn(input=outputs, target=labels)

                # compute the gradient for each variable
                loss.backward()

                # update the weight
                self.optimizer.step()

                # compute the used time
                time_meter.update(time.time() - start_ts)

                # Process display on screen
                if (i + 1) % self.cfg["training"]["print_interval"] == 0:
                    fmt_str = "Iter [{:d}/{:d}]  Loss: {:.4f}  Time/Image: {:.4f}"
                    print_str = fmt_str.format(
                        i + 1,
                        self.cfg["training"]["train_iters"],
                        loss.item(),
                        time_meter.avg / self.cfg["training"]["batch_size"],
                    )
                    print(print_str)
                    self.logger.info(print_str)
                    self.writer.add_scalar("loss/train_loss", loss.item(), i + 1)
                    time_meter.reset()



                #  Validation (During training)
                if i % self.cfg["training"]["val_interval"] == 0 or i == self.cfg["training"]["train_iters"]:
                    self.model.eval()
                    with torch.no_grad():
                        total = 0
                        correct_when = 0
                        correct_who = 0
                        for i_val, data_list in tqdm(enumerate(self.valloader)):

                            if self.if_commun_label != 'None':
                                images_val_list, labels_val_list, commun_label = data_list
                            else:
                                images_val_list, labels_val_list = data_list

                            labels_val = labels_val_list[0]
                            images_val = torch.cat(tuple(images_val_list), dim=1)

                            labels_val = labels_val.to(self.device)
                            if self.if_commun_label != 'None':
                                commun_label = commun_label.to(self.device)

                            images_val = images_val.to(self.device)
                            gt = labels_val.data.cpu().numpy()

                            # image loss
                            outputs, _, action_argmax = self.model(images_val, training=True)
                            val_loss = self.loss_fn(input=outputs, target=labels_val)
                            pred = outputs.data.max(1)[1].cpu().numpy()
                            action_argmax = torch.squeeze(action_argmax)

                            # compute action accuracy
                            if self.if_commun_label != 'None':
                                self.running_metrics_val.update_div(self.if_commun_label, gt, pred, commun_label)
                                self.running_metrics_val.update_selection(self.if_commun_label, commun_label, action_argmax)

                            self.running_metrics_val.update(gt, pred)
                            val_loss_meter.update(val_loss.item())

                    if self.if_commun_label != 'None':
                        when2com_acc, who2com_acc = self.running_metrics_val.get_selection_accuracy()
                        print('Validation when2com accuracy:{}'.format(when2com_acc))
                        print('Validation who2com accuracy:{}'.format(who2com_acc))
                        self.writer.add_scalar("val_metrics/when_com_accuacy", when2com_acc, i)
                        self.writer.add_scalar("val_metrics/who_com_accuracy", who2com_acc, i)
                    else:
                        when2com_acc = 0
                        who2com_acc = 0

                    self.writer.add_scalar("loss/val_loss", val_loss_meter.avg, i)
                    self.logger.info("Iter %d Loss: %.4f" % (i, val_loss_meter.avg))

                    scorers = [self.running_metrics_val]

                    for idx, scorer in enumerate(scorers):
                        score, class_iou = scorer.get_scores()
                        for k, v in score.items():
                            self.logger.info("{}: {}".format(k, v))
                            self.writer.add_scalar("head_{}_val_metrics/{}".format(idx, k), v, i)

                        for k, v in class_iou.items():
                            self.logger.info("{}: {}".format(k, v))
                            self.writer.add_scalar("head_{}_val_metrics/cls_{}".format(idx, k), v, i)


                    # print
                    print('Normal')
                    score, class_iou = self.running_metrics_val.get_only_normal_scores()
                    self.running_metrics_val.print_score(self.n_classes, score, class_iou)

                    print('Noise')
                    score, class_iou = self.running_metrics_val.get_only_noise_scores()
                    self.running_metrics_val.print_score(self.n_classes, score, class_iou)

                    print("Overall")
                    score, class_iou = self.running_metrics_val.get_scores()
                    self.running_metrics_val.print_score(self.n_classes, score, class_iou)

                    val_loss_meter.reset()
                    self.running_metrics_val.reset()

                    if score["Mean IoU : \t"] >= best_iou:
                        best_iou = score["Mean IoU : \t"]
                        state = {
                            "epoch": i,
                            "model_state": self.model.state_dict(),
                            "optimizer_state": self.optimizer.state_dict(),
                            "scheduler_state": self.scheduler.state_dict(),
                            "best_iou": best_iou,
                        }
                        save_path = os.path.join(
                            self.writer.file_writer.get_logdir(),
                            "{}_{}_best_model.pkl".format(self.cfg["model"]["arch"], self.cfg["data"]["dataset"]),
                        )
                        torch.save(state, save_path)
                if i == self.cfg["training"]["train_iters"]:
                    flag = False
                    break
        return save_path

    def load_weight(self, model_path):
        state = convert_state_dict(torch.load(model_path)["model_state"])
        self.model.load_state_dict(state, strict=False)


    def evaluate(self, testloader, inference_mode='activated'): # "val_split"
        running_metrics = runningScore(self.n_classes)

        # Setup Model
        self.model.eval()
        self.model.to(self.device)


        for i, data_list in enumerate(testloader):

            if self.if_commun_label:
                images_list, labels_list, commun_label = data_list
                commun_label = commun_label.to(self.device)
            else:
                images_list, labels_list = data_list

            # multi-view inputs
            images = torch.cat(tuple(images_list), dim=1)

            # multiple output
            if self.MO_flag:
                labels = torch.cat(tuple(labels_list), dim=0)
            else:  # single output
                labels = labels_list[0]

            images = images.to(self.device)
            outputs, _, action_argmax, _ = self.model(images, training=False, inference=inference_mode)

            pred = outputs.data.max(1)[1].cpu().numpy()
            gt = labels.numpy()

            # measurement results
            running_metrics.update(gt, pred)

            if self.if_commun_label:
                running_metrics.update_div(self.if_commun_label, gt, pred, commun_label)


        print('Normal')
        score, class_iou = running_metrics.get_only_normal_scores()
        running_metrics.print_score(self.n_classes, score, class_iou)

        print('Noise')
        score, class_iou = running_metrics.get_only_noise_scores()
        running_metrics.print_score(self.n_classes, score, class_iou)

        print("Overall")
        score, class_iou = running_metrics.get_scores()
        running_metrics.print_score(self.n_classes, score, class_iou)

        return score, class_iou

class Trainer_LearnWho2Com(object):
    def __init__(self, cfg, writer, logger, model, loss_fn, trainloader, valloader, optimizer, scheduler, device):

        self.cfg = cfg
        self.writer = writer
        self.logger = logger
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.loss_fn = loss_fn
        self.MO_flag = self.cfg['model']['multiple_output']
        self.n_classes = 11

        self.running_metrics_val = runningScore(self.n_classes)
        self.device = device


        # some datasets have no labels for communication
        if 'commun_label' in self.cfg["data"]:
            self.if_commun_label = cfg["data"]['commun_label']
        else:
            self.if_commun_label = 'None'

    def train(self):
        start_iter = 0
        print('learnwho2com trainer')
        # resume the training
        if self.cfg["training"]["resume"] is not None:
            if os.path.isfile(self.cfg["training"]["resume"]):
                self.logger.info(
                    "Loading model and optimizer from checkpoint '{}'".format(cfg["training"]["resume"])
                )
                checkpoint = torch.load(self.cfg["training"]["resume"])
                self.model.load_state_dict(checkpoint["model_state"])
                self.optimizer.load_state_dict(checkpoint["optimizer_state"])
                self.scheduler.load_state_dict(checkpoint["scheduler_state"])
                start_iter = checkpoint["epoch"]
                self.logger.info(
                    "Loaded checkpoint '{}' (iter {})".format(
                        self.cfg["training"]["resume"], checkpoint["epoch"]
                    )
                )
            else:
                self.logger.info("No checkpoint found at '{}'".format(self.cfg["training"]["resume"]))


        val_loss_meter = averageMeter()
        time_meter = averageMeter()

        best_iou = -100.0
        i = start_iter
        flag = True

        # Training
        while i <= self.cfg["training"]["train_iters"] and flag:
            for data_list in self.trainloader:
                # iteration timer
                i += 1

                # load data from dataloader
                if self.if_commun_label != 'None':
                    images_list, labels_list, commun_label = data_list
                else:
                    images_list, labels_list = data_list

                # image and labels list 2 tensor
                labels = labels_list[0]
                images = torch.cat(tuple(images_list), dim=1)

                # timer started
                start_ts = time.time()

                self.scheduler.step()
                self.model.train()  # matters for batchnorm/dropout

                # from cpu to gpu
                images = images.to(self.device)
                labels = labels.to(self.device)
                if self.if_commun_label != 'None':
                    commun_label = commun_label.to(self.device)

                # clean the optimizer
                self.optimizer.zero_grad()

                # model inference
                outputs, log_action, action_argmax = self.model(images, training=True)

                # compute loss
                loss = self.loss_fn(input=outputs, target=labels)

                # compute the gradient for each variable
                loss.backward()

                # update the weight
                self.optimizer.step()

                # compute the used time
                time_meter.update(time.time() - start_ts)

                # Process display on screen
                if (i + 1) % self.cfg["training"]["print_interval"] == 0:
                    fmt_str = "Iter [{:d}/{:d}]  Loss: {:.4f}  Time/Image: {:.4f}"
                    print_str = fmt_str.format(
                        i + 1,
                        self.cfg["training"]["train_iters"],
                        loss.item(),
                        time_meter.avg / self.cfg["training"]["batch_size"],
                    )
                    print(print_str)
                    self.logger.info(print_str)
                    self.writer.add_scalar("loss/train_loss", loss.item(), i + 1)
                    time_meter.reset()



                #  Validation (During training)
                if i % self.cfg["training"]["val_interval"] == 0 or i == self.cfg["training"]["train_iters"]:
                    self.model.eval()
                    with torch.no_grad():
                        total = 0
                        correct_when = 0
                        correct_who = 0
                        for i_val, data_list in tqdm(enumerate(self.valloader)):

                            if self.if_commun_label != 'None':
                                images_val_list, labels_val_list, commun_label = data_list
                            else:
                                images_val_list, labels_val_list = data_list

                            labels_val = labels_val_list[0]
                            images_val = torch.cat(tuple(images_val_list), dim=1)

                            labels_val = labels_val.to(self.device)
                            if self.if_commun_label != 'None':
                                commun_label = commun_label.to(self.device)

                            images_val = images_val.to(self.device)
                            gt = labels_val.data.cpu().numpy()

                            # image loss
                            outputs, _, action_argmax = self.model(images_val, training=True)
                            val_loss = self.loss_fn(input=outputs, target=labels_val)
                            pred = outputs.data.max(1)[1].cpu().numpy()
                            action_argmax = torch.squeeze(action_argmax)

                            # compute action accuracy
                            if self.if_commun_label != 'None':
                                self.running_metrics_val.update_div(self.if_commun_label, gt, pred, commun_label)
                                self.running_metrics_val.update_selection(self.if_commun_label, commun_label, action_argmax + 1)
                                    # plus one since target is not included in "alwaysCom" model

                            self.running_metrics_val.update(gt, pred)
                            val_loss_meter.update(val_loss.item())

                    if self.if_commun_label != 'None':
                        when2com_acc, who2com_acc = self.running_metrics_val.get_selection_accuracy()
                        print('Validation when2com accuracy:{}'.format(when2com_acc))
                        print('Validation who2com accuracy:{}'.format(who2com_acc))
                        self.writer.add_scalar("val_metrics/when_com_accuacy", when2com_acc, i)
                        self.writer.add_scalar("val_metrics/who_com_accuracy", who2com_acc, i)
                    else:
                        when2com_acc = 0
                        who2com_acc = 0

                    # for tensorboard
                    self.writer.add_scalar("loss/val_loss", val_loss_meter.avg, i)
                    self.logger.info("Iter %d Loss: %.4f" % (i, val_loss_meter.avg))

                    scorers = [self.running_metrics_val]

                    for idx, scorer in enumerate(scorers):
                        score, class_iou = scorer.get_scores()
                        for k, v in score.items():
                            self.logger.info("{}: {}".format(k, v))
                            self.writer.add_scalar("head_{}_val_metrics/{}".format(idx, k), v, i)

                        for k, v in class_iou.items():
                            self.logger.info("{}: {}".format(k, v))
                            self.writer.add_scalar("head_{}_val_metrics/cls_{}".format(idx, k), v, i)


                    # print
                    print('Normal')
                    score, class_iou = self.running_metrics_val.get_only_normal_scores()
                    self.running_metrics_val.print_score(self.n_classes, score, class_iou)

                    print('Noise')
                    score, class_iou = self.running_metrics_val.get_only_noise_scores()
                    self.running_metrics_val.print_score(self.n_classes, score, class_iou)

                    print("Overall")
                    score, class_iou = self.running_metrics_val.get_scores()
                    self.running_metrics_val.print_score(self.n_classes, score, class_iou)

                    val_loss_meter.reset()
                    self.running_metrics_val.reset()

                    if score["Mean IoU : \t"] >= best_iou:
                        best_iou = score["Mean IoU : \t"]
                        state = {
                            "epoch": i,
                            "model_state": self.model.state_dict(),
                            "optimizer_state": self.optimizer.state_dict(),
                            "scheduler_state": self.scheduler.state_dict(),
                            "best_iou": best_iou,
                        }
                        save_path = os.path.join(
                            self.writer.file_writer.get_logdir(),
                            "{}_{}_best_model.pkl".format(self.cfg["model"]["arch"], self.cfg["data"]["dataset"]),
                        )
                        torch.save(state, save_path)
                if i == self.cfg["training"]["train_iters"]:
                    flag = False
                    break
        return save_path

    def load_weight(self, model_path):
        state = convert_state_dict(torch.load(model_path)["model_state"])
        self.model.load_state_dict(state, strict=False)

    def evaluate(self, testloader, inference_mode='argmax_test'): # "val_split"

        running_metrics = runningScore(self.n_classes)

        # Setup Model
        self.model.eval()
        self.model.to(self.device)

        for i, data_list in enumerate(testloader):
            if self.if_commun_label:
                images_list, labels_list, commun_label = data_list
                commun_label = commun_label.to(self.device)
            else:
                images_list, labels_list = data_list

            # multi-view inputs
            images = torch.cat(tuple(images_list), dim=1)

            # multi-view output
            if self.MO_flag:
                labels = torch.cat(tuple(labels_list), dim=0)
            else:  # single output
                labels = labels_list[0]

            images = images.to(self.device)

            # MODEL INFERENCE
            outputs, action, action_argmax = self.model(images, training=False, inference=inference_mode)
            pred = outputs.data.max(1)[1].cpu().numpy()
            gt = labels.numpy()

            # measurement results
            running_metrics.update(gt, pred)
            if self.if_commun_label:
                running_metrics.update_div(self.if_commun_label, gt, pred, commun_label)
                running_metrics.update_selection(self.if_commun_label, commun_label, action_argmax+1)

        if self.if_commun_label:
            when2com_acc, who2com_acc = running_metrics.get_selection_accuracy()
            print('Validation when2com accuracy:{}'.format(when2com_acc))
            print('Validation who2com accuracy:{}'.format(who2com_acc))
        else:
            when2com_acc = 0
            who2com_acc = 0


        print('Normal')
        score, class_iou = running_metrics.get_only_normal_scores()
        running_metrics.print_score(self.n_classes, score, class_iou)

        print('Noise')
        score, class_iou = running_metrics.get_only_noise_scores()
        running_metrics.print_score(self.n_classes, score, class_iou)

        print("Overall")
        score, class_iou = running_metrics.get_scores()
        running_metrics.print_score(self.n_classes, score, class_iou)

        return score, class_iou

class Trainer_MIMOcom(object):
    def __init__(self, cfg, writer, logger, model, loss_fn, trainloader, valloader, optimizer, scheduler, device):

        self.cfg = cfg
        self.writer = writer
        self.logger = logger
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.n_classes = 11
        self.loss_fn = loss_fn


        self.running_metrics_val = runningScore(self.n_classes)
        self.device = device
        self.MO_flag = self.cfg['model']['multiple_output']

        # some datasets have no labels for communication
        if 'commun_label' in self.cfg["data"]:
            self.if_commun_label = cfg["data"]['commun_label']
        else:
            self.if_commun_label = 'None'

    def train(self):
        # load model
        print('LearnMIMOCom_Trainer')
        start_iter = 0
        if self.cfg["training"]["resume"] is not None:
            if os.path.isfile(self.cfg["training"]["resume"]):
                self.logger.info(
                    "Loading model and optimizer from checkpoint '{}'".format(self.cfg["training"]["resume"])
                )
                checkpoint = torch.load(self.cfg["training"]["resume"])
                self.model.load_state_dict(checkpoint["model_state"])
                self.optimizer.load_state_dict(checkpoint["optimizer_state"])
                self.scheduler.load_state_dict(checkpoint["scheduler_state"])
                start_iter = checkpoint["epoch"]
                self.logger.info(
                    "Loaded checkpoint '{}' (iter {})".format(
                        self.cfg["training"]["resume"], checkpoint["epoch"]
                    )
                )
            else:
                self.logger.info("No checkpoint found at '{}'".format(self.cfg["training"]["resume"]))

        val_loss_meter = averageMeter()
        time_meter = averageMeter()

        best_iou = -100.0
        i = start_iter
        flag = True

        # training
        while i <= self.cfg["training"]["train_iters"] and flag:
            for data_list in self.trainloader:
                i += 1
                start_ts = time.time()

                if self.if_commun_label != 'None':
                    images_list, labels_list, commun_label = data_list
                else:
                    images_list, labels_list = data_list
                images = torch.cat(tuple(images_list), dim=1)

                if self.MO_flag:  # multiple output
                    labels = torch.cat(tuple(labels_list), dim=0)
                else:  # single output
                    labels = labels_list[0]

                self.scheduler.step()
                self.model.train()  # matters for batchnorm/dropout

                images = images.to(self.device)
                labels = labels.to(self.device)
                if self.if_commun_label != 'None':
                    commun_label = commun_label.to(self.device)


                # image loss
                self.optimizer.zero_grad()
                outputs, log_action, action_argmax, _ = self.model(images, training=True, MO_flag=self.MO_flag)

                loss = self.loss_fn(input=outputs, target=labels)
                loss.backward()
                self.optimizer.step()

                time_meter.update(time.time() - start_ts)
                # Process display on screen
                if (i + 1) % self.cfg["training"]["print_interval"] == 0:
                    fmt_str = "Iter [{:d}/{:d}]  Loss: {:.4f}  Time/Image: {:.4f}"
                    print_str = fmt_str.format(
                        i + 1,
                        self.cfg["training"]["train_iters"],
                        loss.item(),
                        time_meter.avg / self.cfg["training"]["batch_size"],
                    )
                    print(print_str)
                    self.logger.info(print_str)
                    self.writer.add_scalar("loss/train_loss", loss.item(), i + 1)
                    time_meter.reset()

                ###  Validation
                if i % self.cfg["training"]["val_interval"] == 0 or i == self.cfg["training"]["train_iters"]:
                    self.model.eval()
                    with torch.no_grad():
                        for i_val, data_list in tqdm(enumerate(self.valloader)):

                            if self.if_commun_label != 'None':
                                images_val_list, labels_val_list, commun_label = data_list
                                commun_label = commun_label.to(self.device)
                            else:
                                images_val_list, labels_val_list = data_list
                            images_val = torch.cat(tuple(images_val_list), dim=1)

                            if self.MO_flag:  # obtain multiple ground-truth
                                labels_val = torch.cat(tuple(labels_val_list), dim=0)
                            else:  # only select one view gt mask
                                labels_val = labels_val_list[0]

                            labels_val = labels_val.to(self.device)
                            images_val = images_val.to(self.device)
                            gt = labels_val.data.cpu().numpy()

                            # image loss
                            outputs, _, action_argmax, _ = self.model(images_val, training=True, MO_flag=self.MO_flag)
                            val_loss = self.loss_fn(input=outputs, target=labels_val)
                            pred = outputs.data.max(1)[1].cpu().numpy()

                            # compute action accuracy
                            if self.if_commun_label != 'None':
                                self.running_metrics_val.update_div(self.if_commun_label, gt, pred, commun_label)
                                self.running_metrics_val.update_selection(self.if_commun_label, commun_label, action_argmax)

                            self.running_metrics_val.update(gt, pred)
                            val_loss_meter.update(val_loss.item())

                    if self.if_commun_label != 'None':
                        when2com_acc, who2com_acc = self.running_metrics_val.get_selection_accuracy()
                        print('Validation when2com accuracy:{}'.format(when2com_acc))
                        print('Validation who2com accuracy:{}'.format(who2com_acc))
                        self.writer.add_scalar("val_metrics/when_com_accuacy", when2com_acc, i)
                        self.writer.add_scalar("val_metrics/who_com_accuracy", who2com_acc, i)

                    self.writer.add_scalar("loss/val_loss", val_loss_meter.avg, i)
                    self.logger.info("Iter %d Loss: %.4f" % (i, val_loss_meter.avg))

                    print('Normal')
                    score, class_iou = self.running_metrics_val.get_only_normal_scores()
                    self.running_metrics_val.print_score(self.n_classes, score, class_iou)

                    print('Noise')
                    score, class_iou = self.running_metrics_val.get_only_noise_scores()
                    self.running_metrics_val.print_score(self.n_classes, score, class_iou)

                    print("Overall")
                    score, class_iou = self.running_metrics_val.get_scores()
                    self.running_metrics_val.print_score(self.n_classes, score, class_iou)

                    val_loss_meter.reset()
                    self.running_metrics_val.reset()

                    # store the best model
                    if score["Mean IoU : \t"] >= best_iou:
                        best_iou = score["Mean IoU : \t"]
                        state = {
                            "epoch": i,
                            "model_state": self.model.state_dict(),
                            "optimizer_state": self.optimizer.state_dict(),
                            "scheduler_state": self.scheduler.state_dict(),
                            "best_iou": best_iou,
                        }
                        save_path = os.path.join(
                            self.writer.file_writer.get_logdir(),
                            "{}_{}_best_model.pkl".format(self.cfg["model"]["arch"], self.cfg["data"]["dataset"]),
                        )
                        torch.save(state, save_path)
                if i == self.cfg["training"]["train_iters"]:
                    flag = False
                    break
        return save_path

    def load_weight(self, model_path):
        state = convert_state_dict(torch.load(model_path)["model_state"])
        self.model.load_state_dict(state, strict=False)

    def evaluate(self, testloader,inference_mode='activated'): # "val_split"

        running_metrics = runningScore(self.n_classes)

        # Setup Model
        self.model.eval()
        self.model.to(self.device)


        for i, data_list in enumerate(testloader):
            if self.if_commun_label:
                images_list, labels_list, commun_label = data_list
                commun_label = commun_label.to(self.device)
            else:
                images_list, labels_list = data_list

            # multi-view inputs
            images = torch.cat(tuple(images_list), dim=1)


            # multiple output
            if self.MO_flag:
                labels = torch.cat(tuple(labels_list), dim=0)
            else:  # single output
                labels = labels_list[0]

            images = images.to(self.device)
            outputs, _, action_argmax, bandW = self.model(images, training=False, MO_flag=self.MO_flag, inference=inference_mode)


            pred = outputs.data.max(1)[1].cpu().numpy()
            gt = labels.numpy()

            # measurement results
            running_metrics.update(gt, pred)
            running_metrics.update_bandW(bandW)

            if self.if_commun_label:
                running_metrics.update_div(self.if_commun_label, gt, pred, commun_label)
                running_metrics.update_selection(self.if_commun_label, commun_label, action_argmax)

        if self.if_commun_label:
            when2com_acc, who2com_acc = running_metrics.get_selection_accuracy()
            print('Validation when2com accuracy:{}'.format(when2com_acc))
            print('Validation who2com accuracy:{}'.format(who2com_acc))
        else:
            when2com_acc = 0
            who2com_acc = 0


        avg_bandW = running_metrics.get_avg_bandW()
        print('Bandwidth: ' + str(avg_bandW))


        print('Normal')
        score, class_iou = running_metrics.get_only_normal_scores()
        running_metrics.print_score(self.n_classes, score, class_iou)

        print('Noise')
        score, class_iou = running_metrics.get_only_noise_scores()
        running_metrics.print_score(self.n_classes, score, class_iou)

        print("Overall")
        score, class_iou = running_metrics.get_scores()
        running_metrics.print_score(self.n_classes, score, class_iou)

        return score, class_iou

class Trainer_MIMOcomWho(object):
    def __init__(self, cfg, writer, logger, model, loss_fn, trainloader, valloader, optimizer, scheduler, device):

        self.cfg = cfg
        self.writer = writer
        self.logger = logger
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.n_classes = 11
        self.loss_fn = loss_fn

        self.running_metrics_val = runningScore(self.n_classes)
        self.device = device
        self.MO_flag = self.cfg['model']['multiple_output']

        # some datasets have no labels for communication
        if 'commun_label' in self.cfg["data"]:
            self.if_commun_label = cfg["data"]['commun_label']
        else:
            self.if_commun_label = 'None'

    def train(self):
        # load model
        print('LearnMIMOComWho_Trainer')
        start_iter = 0
        if self.cfg["training"]["resume"] is not None:
            if os.path.isfile(self.cfg["training"]["resume"]):
                self.logger.info(
                    "Loading model and optimizer from checkpoint '{}'".format(self.cfg["training"]["resume"])
                )
                checkpoint = torch.load(self.cfg["training"]["resume"])
                self.model.load_state_dict(checkpoint["model_state"])
                self.optimizer.load_state_dict(checkpoint["optimizer_state"])
                self.scheduler.load_state_dict(checkpoint["scheduler_state"])
                start_iter = checkpoint["epoch"]
                self.logger.info(
                    "Loaded checkpoint '{}' (iter {})".format(
                        self.cfg["training"]["resume"], checkpoint["epoch"]
                    )
                )
            else:
                self.logger.info("No checkpoint found at '{}'".format(self.cfg["training"]["resume"]))

        val_loss_meter = averageMeter()
        time_meter = averageMeter()

        best_iou = -100.0
        i = start_iter
        flag = True

        # training
        while i <= self.cfg["training"]["train_iters"] and flag:
            for data_list in self.trainloader:
                i += 1
                start_ts = time.time()

                if self.if_commun_label != 'None':
                    images_list, labels_list, commun_label = data_list
                else:
                    images_list, labels_list = data_list
                images = torch.cat(tuple(images_list), dim=1)

                if self.MO_flag:  # multiple output
                    labels = torch.cat(tuple(labels_list), dim=0)
                else:  # single output
                    labels = labels_list[0]

                self.scheduler.step()
                self.model.train()  # matters for batchnorm/dropout

                images = images.to(self.device)
                labels = labels.to(self.device)
                if self.if_commun_label != 'None':
                    commun_label = commun_label.to(self.device)


                # image loss
                self.optimizer.zero_grad()
                outputs, log_action, action_argmax, _ = self.model(images, training=True, MO_flag=self.MO_flag)

                loss = self.loss_fn(input=outputs, target=labels)
                loss.backward()
                self.optimizer.step()

                time_meter.update(time.time() - start_ts)
                # Process display on screen
                if (i + 1) % self.cfg["training"]["print_interval"] == 0:
                    fmt_str = "Iter [{:d}/{:d}]  Loss: {:.4f}  Time/Image: {:.4f}"
                    print_str = fmt_str.format(
                        i + 1,
                        self.cfg["training"]["train_iters"],
                        loss.item(),
                        time_meter.avg / self.cfg["training"]["batch_size"],
                    )
                    print(print_str)
                    self.logger.info(print_str)
                    self.writer.add_scalar("loss/train_loss", loss.item(), i + 1)
                    time_meter.reset()

                ###  Validation
                if i % self.cfg["training"]["val_interval"] == 0 or i == self.cfg["training"]["train_iters"]:
                    self.model.eval()
                    with torch.no_grad():
                        for i_val, data_list in tqdm(enumerate(self.valloader)):

                            if self.if_commun_label != 'None':
                                images_val_list, labels_val_list, commun_label = data_list
                                commun_label = commun_label.to(self.device)
                            else:
                                images_val_list, labels_val_list = data_list
                            images_val = torch.cat(tuple(images_val_list), dim=1)

                            if self.MO_flag:  # obtain multiple ground-truth
                                labels_val = torch.cat(tuple(labels_val_list), dim=0)
                            else:  # only select one view gt mask
                                labels_val = labels_val_list[0]

                            labels_val = labels_val.to(self.device)
                            images_val = images_val.to(self.device)
                            gt = labels_val.data.cpu().numpy()

                            # image loss
                            outputs, _, action_argmax, _ = self.model(images_val, training=True, MO_flag=self.MO_flag)
                            val_loss = self.loss_fn(input=outputs, target=labels_val)
                            pred = outputs.data.max(1)[1].cpu().numpy()

                            # compute action accuracy
                            if self.if_commun_label != 'None':
                                self.running_metrics_val.update_div(self.if_commun_label, gt, pred, commun_label)
                                self.running_metrics_val.update_selection(self.if_commun_label, commun_label, action_argmax)

                            self.running_metrics_val.update(gt, pred)
                            val_loss_meter.update(val_loss.item())

                    if self.if_commun_label != 'None':
                        when2com_acc, who2com_acc = self.running_metrics_val.get_selection_accuracy()
                        print('Validation when2com accuracy:{}'.format(when2com_acc))
                        print('Validation who2com accuracy:{}'.format(who2com_acc))
                        self.writer.add_scalar("val_metrics/when_com_accuacy", when2com_acc, i)
                        self.writer.add_scalar("val_metrics/who_com_accuracy", who2com_acc, i)

                    self.writer.add_scalar("loss/val_loss", val_loss_meter.avg, i)
                    self.logger.info("Iter %d Loss: %.4f" % (i, val_loss_meter.avg))

                    print('Normal')
                    score, class_iou = self.running_metrics_val.get_only_normal_scores()
                    self.running_metrics_val.print_score(self.n_classes, score, class_iou)

                    print('Noise')
                    score, class_iou = self.running_metrics_val.get_only_noise_scores()
                    self.running_metrics_val.print_score(self.n_classes, score, class_iou)

                    print("Overall")
                    score, class_iou = self.running_metrics_val.get_scores()
                    self.running_metrics_val.print_score(self.n_classes, score, class_iou)

                    val_loss_meter.reset()
                    self.running_metrics_val.reset()

                    # store the best model
                    if score["Mean IoU : \t"] >= best_iou:
                        best_iou = score["Mean IoU : \t"]
                        state = {
                            "epoch": i,
                            "model_state": self.model.state_dict(),
                            "optimizer_state": self.optimizer.state_dict(),
                            "scheduler_state": self.scheduler.state_dict(),
                            "best_iou": best_iou,
                        }
                        save_path = os.path.join(
                            self.writer.file_writer.get_logdir(),
                            "{}_{}_best_model.pkl".format(self.cfg["model"]["arch"], self.cfg["data"]["dataset"]),
                        )
                        torch.save(state, save_path)
                if i == self.cfg["training"]["train_iters"]:
                    flag = False
                    break
        return save_path
    def load_weight(self, model_path):
        state = convert_state_dict(torch.load(model_path)["model_state"])
        self.model.load_state_dict(state, strict=False)

    def evaluate(self, testloader,inference_mode='activated'): # "val_split"

        running_metrics = runningScore(self.n_classes)

        # Setup Model
        self.model.eval()
        self.model.to(self.device)


        for i, data_list in enumerate(testloader):
            if self.if_commun_label:
                images_list, labels_list, commun_label = data_list
                commun_label = commun_label.to(self.device)
            else:
                images_list, labels_list = data_list

            # multi-view inputs
            images = torch.cat(tuple(images_list), dim=1)


            # multiple output
            if self.MO_flag:
                labels = torch.cat(tuple(labels_list), dim=0)
            else:  # single output
                labels = labels_list[0]

            images = images.to(self.device)
            outputs, _, action_argmax, bandW = self.model(images, training=False, MO_flag=self.MO_flag, inference=inference_mode)


            pred = outputs.data.max(1)[1].cpu().numpy()
            gt = labels.numpy()

            # measurement results
            running_metrics.update(gt, pred)
            running_metrics.update_bandW(bandW)

            if self.if_commun_label:
                running_metrics.update_div(self.if_commun_label, gt, pred, commun_label)
                running_metrics.update_selection(self.if_commun_label, commun_label, action_argmax)

        if self.if_commun_label:
            when2com_acc, who2com_acc = running_metrics.get_selection_accuracy()
            print('Validation when2com accuracy:{}'.format(when2com_acc))
            print('Validation who2com accuracy:{}'.format(who2com_acc))
        else:
            when2com_acc = 0
            who2com_acc = 0


        avg_bandW = running_metrics.get_avg_bandW()
        print('Bandwidth: ' + str(avg_bandW))


        print('Normal')
        score, class_iou = running_metrics.get_only_normal_scores()
        running_metrics.print_score(self.n_classes, score, class_iou)

        print('Noise')
        score, class_iou = running_metrics.get_only_noise_scores()
        running_metrics.print_score(self.n_classes, score, class_iou)

        print("Overall")
        score, class_iou = running_metrics.get_scores()
        running_metrics.print_score(self.n_classes, score, class_iou)

        return score, class_iou

class Trainer_MIMO_All_agents(object):
    def __init__(self, cfg, writer, logger, model, loss_fn, trainloader, valloader, optimizer, scheduler, device):

        self.cfg = cfg
        self.writer = writer
        self.logger = logger
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.n_classes = 11
        self.loss_fn = loss_fn

        self.running_metrics_val = runningScore(self.n_classes)
        self.device = device
        self.MO_flag = self.cfg['model']['multiple_output']
        if 'commun_label' in self.cfg["data"]:
            self.if_commun_label = cfg["data"]['commun_label']
        else:
            self.if_commun_label = 'None'

    def train(self):
        print('MIMO_All_Agent_Trainer')
        start_iter = 0
        if self.cfg["training"]["resume"] is not None:
            if os.path.isfile(self.cfg["training"]["resume"]):
                self.logger.info(
                    "Loading model and optimizer from checkpoint '{}'".format(self.cfg["training"]["resume"])
                )
                checkpoint = torch.load(self.cfg["training"]["resume"])
                self.model.load_state_dict(checkpoint["model_state"])
                self.optimizer.load_state_dict(checkpoint["optimizer_state"])
                self.scheduler.load_state_dict(checkpoint["scheduler_state"])
                start_iter = checkpoint["epoch"]
                self.logger.info(
                    "Loaded checkpoint '{}' (iter {})".format(
                        self.cfg["training"]["resume"], checkpoint["epoch"]
                    )
                )
            else:
                self.logger.info("No checkpoint found at '{}'".format(self.cfg["training"]["resume"]))

        val_loss_meter = averageMeter()
        time_meter = averageMeter()

        best_iou = -100.0
        i = start_iter
        flag = True

        while i <= self.cfg["training"]["train_iters"] and flag:
            for data_list in self.trainloader:

                if self.if_commun_label != 'None':
                    images_list, labels_list, commun_label = data_list
                else:
                    images_list, labels_list = data_list

                # only first image
                images = images_list[0]
                labels = labels_list[0]

                images_list[0] = images
                images = torch.cat(tuple(images_list), dim=1)

                if self.cfg['model']['multiple_output']:
                    labels = torch.cat(tuple(labels_list), dim=0)

                i += 1
                start_ts = time.time()
                self.scheduler.step()
                self.model.train()  # matters for batchnorm/dropout

                images = images.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()

                if self.cfg['model']['shuffle_features'] == 'selection':
                    outputs, rand_action = self.model(images)
                else:
                    outputs = self.model(images)

                loss = self.loss_fn(input=outputs, target=labels)
                loss.backward()
                self.optimizer.step()

                time_meter.update(time.time() - start_ts)

                # Process display on screen
                if i % self.cfg["training"]["print_interval"] == 0:
                    fmt_str = "Iter [{:d}/{:d}]  Loss: {:.4f}  Time/Image: {:.4f}"
                    print_str = fmt_str.format(
                        i,
                        self.cfg["training"]["train_iters"],
                        loss.item(),
                        time_meter.avg / self.cfg["training"]["batch_size"],
                    )

                    print(print_str)
                    self.logger.info(print_str)
                    self.writer.add_scalar("loss/train_loss", loss.item(), i)
                    time_meter.reset()

                #  Validation
                if (i) % self.cfg["training"]["val_interval"] == 0 or (i) == self.cfg["training"]["train_iters"]:
                    self.model.eval()
                    with torch.no_grad():
                        for i_val, (data_list) in tqdm(enumerate(self.valloader), ncols=20,
                                                                              desc='Validation'):

                            if self.if_commun_label != 'None':
                                images_val_list, labels_val_list, commun_label = data_list
                            else:
                                images_val_list, labels_val_list = data_list



                            images_val = torch.cat(tuple(images_val_list), dim=1)
                            labels_val = labels_val_list[0]

                            if self.cfg['model']['multiple_output']:  # mimo single
                                labels_val = torch.cat(tuple(labels_val_list), dim=0)

                            images_val = images_val.to(self.device)
                            labels_val = labels_val.to(self.device)

                            if self.cfg['model']['shuffle_features'] == 'selection':
                                outputs, rand_action = self.model(images_val)
                            else:
                                outputs = self.model(images_val)

                            val_loss = self.loss_fn(input=outputs, target=labels_val)

                            pred = outputs.data.max(1)[1].cpu().numpy()
                            gt = labels_val.data.cpu().numpy()

                            self.running_metrics_val.update(gt, pred)
                            val_loss_meter.update(val_loss.item())

                    self.writer.add_scalar("loss/val_loss", val_loss_meter.avg, i)
                    self.logger.info("Iter %d Loss: %.4f" % (i, val_loss_meter.avg))

                    score, class_iou = self.running_metrics_val.get_scores()
                    for k, v in score.items():
                        print(k, v)
                        self.logger.info("{}: {}".format(k, v))
                        self.writer.add_scalar("val_metrics/{}".format(k), v, i)

                    for k, v in class_iou.items():
                        self.logger.info("{}: {}".format(k, v))
                        self.writer.add_scalar("val_metrics/cls_{}".format(k), v, i)

                    val_loss_meter.reset()
                    self.running_metrics_val.reset()

                    if score["Mean IoU : \t"] >= best_iou:
                        best_iou = score["Mean IoU : \t"]
                        state = {
                            "epoch": i,
                            "model_state": self.model.state_dict(),
                            "optimizer_state": self.optimizer.state_dict(),
                            "scheduler_state": self.scheduler.state_dict(),
                            "best_iou": best_iou,
                        }
                        save_path = os.path.join(
                            self.writer.file_writer.get_logdir(),
                            "{}_{}_best_model.pkl".format(self.cfg["model"]["arch"], self.cfg["data"]["dataset"]),
                        )
                        torch.save(state, save_path)

                if i == self.cfg["training"]["train_iters"]:
                    flag = False
                    break
        return save_path
    
    def load_weight(self, model_path):
        state = convert_state_dict(torch.load(model_path)["model_state"])
        self.model.load_state_dict(state, strict=False)


    def evaluate(self, testloader): # "val_split"
        running_metrics = runningScore(self.n_classes)

        # Setup Model
        self.model.eval()
        self.model.to(self.device)

        for i, data_list in enumerate(testloader):
            if self.if_commun_label:
                images_list, labels_list, commun_label = data_list
                commun_label = commun_label.to(self.device)
            else:
                images_list, labels_list = data_list

            # multi-view inputs
            images = torch.cat(tuple(images_list), dim=1)

            # multi-view output
            if self.MO_flag:
                labels = torch.cat(tuple(labels_list), dim=0)
            else:  # single output
                labels = labels_list[0]

            images = images.to(self.device)

            # MODEL INFERENCE
            if self.cfg['model']['shuffle_features'] == 'selection':
                outputs, action_argmax = self.model(images)
            else:
                outputs = self.model(images)

            pred = outputs.data.max(1)[1].cpu().numpy()
            gt = labels.numpy()

            # measurement results
            running_metrics.update(gt, pred)

            if self.if_commun_label:
                running_metrics.update_div(self.if_commun_label, gt, pred, commun_label)


        print('Normal')
        score, class_iou = running_metrics.get_only_normal_scores()
        running_metrics.print_score(self.n_classes, score, class_iou)

        print('Noise')
        score, class_iou = running_metrics.get_only_noise_scores()
        running_metrics.print_score(self.n_classes, score, class_iou)

        print("Overall")
        score, class_iou = running_metrics.get_scores()
        running_metrics.print_score(self.n_classes, score, class_iou)

        return score, class_iou

class Trainer_Single_agent(object):
    def __init__(self, cfg, writer, logger, model, loss_fn, trainloader, valloader, optimizer, scheduler, device):

        self.cfg = cfg
        self.writer = writer
        self.logger = logger
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader

        self.optimizer = optimizer
        self.scheduler = scheduler

        self.n_classes = 11
        self.loss_fn = loss_fn

        self.running_metrics_val = runningScore(self.n_classes)
        self.device = device
        self.MO_flag = self.cfg['model']['multiple_output']

    def train(self):
        print('Training')
        start_iter = 0
        if self.cfg["training"]["resume"] is not None:
            if os.path.isfile(self.cfg["training"]["resume"]):
                self.logger.info(
                    "Loading model and optimizer from checkpoint '{}'".format(self.cfg["training"]["resume"])
                )
                checkpoint = torch.load(self.cfg["training"]["resume"])
                self.model.load_state_dict(checkpoint["model_state"])
                self.optimizer.load_state_dict(checkpoint["optimizer_state"])
                self.scheduler.load_state_dict(checkpoint["scheduler_state"])
                start_iter = checkpoint["epoch"]
                self.logger.info(
                    "Loaded checkpoint '{}' (iter {})".format(
                        self.cfg["training"]["resume"], checkpoint["epoch"]
                    )
                )
            else:
                self.logger.info("No checkpoint found at '{}'".format(self.cfg["training"]["resume"]))

        val_loss_meter = averageMeter()
        time_meter = averageMeter()

        best_iou = -100.0
        i = start_iter
        flag = True

        while i <= self.cfg["training"]["train_iters"] and flag:
            for data_list in self.trainloader:

                images_list, labels_list = data_list

                # only first image
                images = images_list[0]
                labels = labels_list[0]

                if self.cfg['model']['multiple_output']:  # mimo single
                    labels = torch.cat(tuple(labels_list), dim=0)
                    images = torch.cat(tuple(images_list), dim=0)

                i += 1
                start_ts = time.time()
                self.scheduler.step()
                self.model.train()  # matters for batchnorm/dropout

                images = images.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()

                outputs = self.model(images)

                loss = self.loss_fn(input=outputs, target=labels)
                loss.backward()
                self.optimizer.step()

                time_meter.update(time.time() - start_ts)

                # Process display on screen
                if i % self.cfg["training"]["print_interval"] == 0:
                    fmt_str = "Iter [{:d}/{:d}]  Loss: {:.4f}  Time/Image: {:.4f}"
                    print_str = fmt_str.format(
                        i,
                        self.cfg["training"]["train_iters"],
                        loss.item(),
                        time_meter.avg / self.cfg["training"]["batch_size"],
                    )

                    print(print_str)
                    self.logger.info(print_str)
                    self.writer.add_scalar("loss/train_loss", loss.item(), i)
                    time_meter.reset()

                # Validation
                if i % self.cfg["training"]["val_interval"] == 0 or i == self.cfg["training"]["train_iters"]:
                    self.model.eval()
                    with torch.no_grad():
                        for i_val, (images_val_list, labels_val_list) in tqdm(enumerate(self.valloader), ncols=20,
                                                                              desc='Validation'):
                            images_val = images_val_list[0]
                            labels_val = labels_val_list[0]

                            if self.cfg['model']['multiple_output']:  # mimo single
                                labels_val = torch.cat(tuple(labels_val_list), dim=0)
                                if self.cfg["model"]["arch"] == 'Single_agent':
                                    images_val = torch.cat(tuple(images_val_list), dim=0)

                            images_val = images_val.to(self.device)
                            labels_val = labels_val.to(self.device)

                            outputs = self.model(images_val)

                            val_loss = self.loss_fn(input=outputs, target=labels_val)

                            pred = outputs.data.max(1)[1].cpu().numpy()
                            gt = labels_val.data.cpu().numpy()

                            self.running_metrics_val.update(gt, pred)
                            val_loss_meter.update(val_loss.item())

                    self.writer.add_scalar("loss/val_loss", val_loss_meter.avg, i)
                    self.logger.info("Iter %d Loss: %.4f" % (i, val_loss_meter.avg))

                    score, class_iou = self.running_metrics_val.get_scores()
                    for k, v in score.items():
                        print(k, v)
                        self.logger.info("{}: {}".format(k, v))
                        self.writer.add_scalar("val_metrics/{}".format(k), v, i)

                    for k, v in class_iou.items():
                        self.logger.info("{}: {}".format(k, v))
                        self.writer.add_scalar("val_metrics/cls_{}".format(k), v, i)

                    val_loss_meter.reset()
                    self.running_metrics_val.reset()

                    if score["Mean IoU : \t"] >= best_iou:
                        best_iou = score["Mean IoU : \t"]
                        state = {
                            "epoch": i,
                            "model_state": self.model.state_dict(),
                            "optimizer_state": self.optimizer.state_dict(),
                            "scheduler_state": self.scheduler.state_dict(),
                            "best_iou": best_iou,
                        }
                        save_path = os.path.join(
                            self.writer.file_writer.get_logdir(),
                            "{}_{}_best_model.pkl".format(self.cfg["model"]["arch"], self.cfg["data"]["dataset"]),
                        )
                        torch.save(state, save_path)

                if i == self.cfg["training"]["train_iters"]:
                    flag = False
                    break
        return save_path


    def load_weight(self, model_path):
        state = convert_state_dict(torch.load(model_path)["model_state"])
        self.model.load_state_dict(state, strict=False)


    def evaluate(self, testloader):

        # local evalutation metric
        running_metrics = runningScore(self.n_classes)

        # Setup Model for evaluaton model
        self.model.eval()
        self.model.to(self.device)

        for i, data_list in enumerate(testloader):
            
            images_list, labels_list = data_list

            # multi-view inputs
            images = torch.cat(tuple(images_list), dim=0)

            # multi-view output
            if self.cfg['model']['multiple_output']:
                labels = torch.cat(tuple(labels_list), dim=0)
            else:  # single output
                labels = labels_list[0]

            images = images.to(self.device)
            outputs = self.model(images)

            pred = outputs.data.max(1)[1].cpu().numpy()
            gt = labels.numpy()

            # measurement results
            running_metrics.update(gt, pred)

        print("Overall")
        score, class_iou = running_metrics.get_scores()
        running_metrics.print_score(self.n_classes, score, class_iou)

        return score, class_iou

class Trainer_All_agents(object):
    def __init__(self, cfg, writer, logger, model, loss_fn, trainloader, valloader, optimizer, scheduler, device):

        self.cfg = cfg
        self.writer = writer
        self.logger = logger
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.n_classes = 11
        self.loss_fn = loss_fn

        self.running_metrics_val = runningScore(self.n_classes)
        self.device = device
        self.MO_flag = self.cfg['model']['multiple_output']
        if 'commun_label' in self.cfg["data"]:
            self.if_commun_label = cfg["data"]['commun_label']
        else:
            self.if_commun_label = 'None'

    def train(self):
        print('Training')
        start_iter = 0
        if self.cfg["training"]["resume"] is not None:
            if os.path.isfile(self.cfg["training"]["resume"]):
                self.logger.info(
                    "Loading model and optimizer from checkpoint '{}'".format(self.cfg["training"]["resume"])
                )
                checkpoint = torch.load(self.cfg["training"]["resume"])
                self.model.load_state_dict(checkpoint["model_state"])
                self.optimizer.load_state_dict(checkpoint["optimizer_state"])
                self.scheduler.load_state_dict(checkpoint["scheduler_state"])
                start_iter = checkpoint["epoch"]
                self.logger.info(
                    "Loaded checkpoint '{}' (iter {})".format(
                        self.cfg["training"]["resume"], checkpoint["epoch"]
                    )
                )
            else:
                self.logger.info("No checkpoint found at '{}'".format(self.cfg["training"]["resume"]))

        val_loss_meter = averageMeter()
        time_meter = averageMeter()

        best_iou = -100.0
        i = start_iter
        flag = True

        while i <= self.cfg["training"]["train_iters"] and flag:
            for data_list in self.trainloader:

                if self.if_commun_label != 'None':
                    images_list, labels_list, commun_label = data_list
                else:
                    images_list, labels_list = data_list

                images = torch.cat(tuple(images_list), dim=1)

                labels = labels_list[0]
                if self.cfg['model']['multiple_output']: # multiple output
                    labels = torch.cat(tuple(labels_list), dim=0)

                i += 1
                start_ts = time.time()
                self.scheduler.step()
                self.model.train()  

                images = images.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()

                if self.cfg['model']['shuffle_features'] == 'selection': # randcom
                    outputs, rand_action = self.model(images)
                else: # catall
                    outputs = self.model(images)

                loss = self.loss_fn(input=outputs, target=labels)
                loss.backward()
                self.optimizer.step()

                time_meter.update(time.time() - start_ts)

                # Process display on screen
                if i % self.cfg["training"]["print_interval"] == 0:
                    fmt_str = "Iter [{:d}/{:d}]  Loss: {:.4f}  Time/Image: {:.4f}"
                    print_str = fmt_str.format(
                        i,
                        self.cfg["training"]["train_iters"],
                        loss.item(),
                        time_meter.avg / self.cfg["training"]["batch_size"],
                    )

                    print(print_str)
                    self.logger.info(print_str)
                    self.writer.add_scalar("loss/train_loss", loss.item(), i)
                    time_meter.reset()

                #  Validation
                if i % self.cfg["training"]["val_interval"] == 0 or i == self.cfg["training"]["train_iters"]:
                    self.model.eval()
                    with torch.no_grad():
                        for i_val, (data_val_list) in tqdm(enumerate(self.valloader), ncols=20,
                                                                              desc='Validation'):


                            if self.if_commun_label != 'None':
                                images_val_list, labels_val_list, commun_label = data_list
                            else:
                                images_val_list, labels_val_list = data_list

                            images_val = torch.cat(tuple(images_val_list), dim=1)

                            labels_val = labels_val_list[0]
                            if self.cfg['model']['multiple_output']: 
                                labels_val = torch.cat(tuple(labels_val_list), dim=0)

                            images_val = images_val.to(self.device)
                            labels_val = labels_val.to(self.device)

                            if self.cfg['model']['shuffle_features'] == 'selection':
                                outputs, rand_action = self.model(images_val)
                            else:
                                outputs = self.model(images_val)

                            val_loss = self.loss_fn(input=outputs, target=labels_val)

                            pred = outputs.data.max(1)[1].cpu().numpy()
                            gt = labels_val.data.cpu().numpy()

                            self.running_metrics_val.update(gt, pred)
                            val_loss_meter.update(val_loss.item())

                    self.writer.add_scalar("loss/val_loss", val_loss_meter.avg, i)
                    self.logger.info("Iter %d Loss: %.4f" % (i, val_loss_meter.avg))

                    score, class_iou = self.running_metrics_val.get_scores()
                    for k, v in score.items():
                        print(k, v)
                        self.logger.info("{}: {}".format(k, v))
                        self.writer.add_scalar("val_metrics/{}".format(k), v, i)

                    for k, v in class_iou.items():
                        self.logger.info("{}: {}".format(k, v))
                        self.writer.add_scalar("val_metrics/cls_{}".format(k), v, i)

                    val_loss_meter.reset()
                    self.running_metrics_val.reset()

                    if score["Mean IoU : \t"] >= best_iou:
                        best_iou = score["Mean IoU : \t"]
                        state = {
                            "epoch": i,
                            "model_state": self.model.state_dict(),
                            "optimizer_state": self.optimizer.state_dict(),
                            "scheduler_state": self.scheduler.state_dict(),
                            "best_iou": best_iou,
                        }
                        save_path = os.path.join(
                            self.writer.file_writer.get_logdir(),
                            "{}_{}_best_model.pkl".format(self.cfg["model"]["arch"], self.cfg["data"]["dataset"]),
                        )
                        torch.save(state, save_path)

                if i == self.cfg["training"]["train_iters"]:
                    flag = False
                    break
        return save_path

    def load_weight(self, model_path):
        state = convert_state_dict(torch.load(model_path)["model_state"])
        self.model.load_state_dict(state, strict=False)

    def evaluate(self, testloader): # "val_split"

        running_metrics = runningScore(self.n_classes)

        # Setup Model
        self.model.eval()
        self.model.to(self.device)


        for i, data_list in enumerate(testloader):
            if self.if_commun_label:
                images_list, labels_list, commun_label = data_list
                commun_label = commun_label.to(self.device)
            else:
                images_list, labels_list = data_list

            # multi-view inputs
            images = torch.cat(tuple(images_list), dim=1)

            # multi-view output
            if self.MO_flag:
                labels = torch.cat(tuple(labels_list), dim=0)
            else:  # single output
                labels = labels_list[0]

            images = images.to(self.device)

            if self.cfg['model']['shuffle_features'] == 'selection':
                outputs, rand_action = self.model(images)
            else:
                outputs = self.model(images)

            pred = outputs.data.max(1)[1].cpu().numpy()
            gt = labels.numpy()

            # measurement results
            running_metrics.update(gt, pred)
            if self.if_commun_label:
                running_metrics.update_div(self.if_commun_label, gt, pred, commun_label)


        print('Normal')
        score, class_iou = running_metrics.get_only_normal_scores()
        running_metrics.print_score(self.n_classes, score, class_iou)

        print('Noise')
        score, class_iou = running_metrics.get_only_noise_scores()
        running_metrics.print_score(self.n_classes, score, class_iou)

        print("Overall")
        score, class_iou = running_metrics.get_scores()
        running_metrics.print_score(self.n_classes, score, class_iou)

        return score, class_iou
