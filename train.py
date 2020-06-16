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
from ptsemseg.metrics import runningScore
from ptsemseg.augmentations import get_composed_augmentations
from ptsemseg.schedulers import get_scheduler
from ptsemseg.optimizers import get_optimizer
from ptsemseg.utils import convert_state_dict

from ptsemseg.trainer import *

from tensorboardX import SummaryWriter


# main function 
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="configs/your_configs.yml",
        help="Configuration file to use",
    )

    parser.add_argument(
        "--gpu",
        nargs="?",
        type=str,
        default="0",
        help="Used GPUs",
    )

    parser.add_argument(
        "--run_time",
        nargs="?",
        type=int,
        default=1,
        help="run_time",
    )

    args = parser.parse_args()

    # Set the gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    run_times = args.run_time

    with open(args.config) as fp:
        cfg = yaml.load(fp)

    data_splits = ['val_split', 'test_split']

    # initialize for results stats
    score_list = {}
    class_iou_list = {}
    acc_list = {}
    if cfg['model']['arch'] == 'LearnWho2Com':
        for infer in ['softmax', 'argmax_test']:
            score_list[infer] = {}
            class_iou_list[infer] = {}
            acc_list[infer] = {}
            for data_sp in data_splits:
                score_list[infer][data_sp] = []
                class_iou_list[infer][data_sp] = []
                acc_list[infer][data_sp] = []
    elif cfg['model']['arch'] == 'LearnWhen2Com' or \
            cfg['model']['arch'] == 'MIMOcom' or \
            cfg['model']['arch'] == 'MIMOcomMultiWarp' or \
            cfg['model']['arch'] == 'MIMOcomWho' :
        for infer in ['softmax', 'argmax_test', 'activated']:
            score_list[infer] = {}
            class_iou_list[infer] = {}
            acc_list[infer] = {}
            for data_sp in data_splits:
                score_list[infer][data_sp] = []
                class_iou_list[infer][data_sp] = []
                acc_list[infer][data_sp] = []
    elif cfg['model']['arch'] == 'Single_agent' or cfg['model']['arch'] == 'All_agents' or cfg['model']['arch'] == 'MIMO_All_agents':
        for infer in ['default']:
            score_list[infer] = {}
            class_iou_list[infer] = {}
            acc_list[infer] = {}
            for data_sp in data_splits:
                score_list[infer][data_sp] = []
                class_iou_list[infer][data_sp] = []
                acc_list[infer][data_sp] = []

    for _ in range(run_times):
        run_id = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')  
        logdir = os.path.join("runs", os.path.basename(args.config)[:-4], str(run_id))
        writer = SummaryWriter(logdir=logdir)

        print("RUNDIR: {}".format(logdir))
        shutil.copy(args.config, logdir)


        # ============= Training =============
        # logger
        logger = get_logger(logdir)
        logger.info("Begin")

        # Setup seeds
        torch.manual_seed(cfg.get("seed", 1337))
        torch.cuda.manual_seed(cfg.get("seed", 1337))
        np.random.seed(cfg.get("seed", 1337))
        random.seed(cfg.get("seed", 1337))

        # Setup device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Setup Dataloader
        data_loader = get_loader(cfg["data"]["dataset"])
        data_path = cfg["data"]["path"]

        # Load communication label (note that some datasets do not provide this)
        if 'commun_label' in cfg["data"]:
            if_commun_label = cfg["data"]['commun_label']
        else:
            if_commun_label = 'None'


        # dataloaders
        t_loader = data_loader(
            data_path,
            is_transform=True,
            split=cfg["data"]["train_split"],
            img_size=(cfg["data"]["img_rows"], cfg["data"]["img_cols"]),
            augmentations=get_composed_augmentations(cfg["training"].get("augmentations", None)),
            target_view=cfg["data"]["target_view"],
            commun_label=if_commun_label
        )

        v_loader = data_loader(
            data_path,
            is_transform=True,
            split=cfg["data"]["val_split"],
            img_size=(cfg["data"]["img_rows"], cfg["data"]["img_cols"]),
            target_view=cfg["data"]["target_view"],
            commun_label=if_commun_label
        )

        trainloader = data.DataLoader(
            t_loader,
            batch_size=cfg["training"]["batch_size"],
            num_workers=cfg["training"]["n_workers"],
            shuffle=True,
            drop_last=True
        )

        valloader = data.DataLoader(
            v_loader,
            batch_size=cfg["training"]["batch_size"],
            num_workers=cfg["training"]["n_workers"]
        )

        # Setup Model
        model = get_model(cfg, t_loader.n_classes).to(device) 
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
        # import pdb; pdb.set_trace()

        # Setup optimizer
        optimizer_cls = get_optimizer(cfg)
        optimizer_params = {k: v for k, v in cfg["training"]["optimizer"].items() if k != "name"}
        optimizer = optimizer_cls(model.parameters(), **optimizer_params)
        logger.info("Using optimizer {}".format(optimizer))

        # Setup scheduler
        scheduler = get_scheduler(optimizer, cfg["training"]["lr_schedule"])

        # Setup loss
        loss_fn = get_loss_function(cfg)
        logger.info("Using loss {}".format(loss_fn))


        # ================== TRAINING ==================
        if cfg['model']['arch'] == 'LearnWhen2Com': # Our when2com
            trainer = Trainer_LearnWhen2Com(cfg, writer, logger, model, loss_fn, trainloader, valloader, optimizer, scheduler, device)
        elif cfg['model']['arch'] == 'LearnWho2Com': # Our who2com
            trainer = Trainer_LearnWho2Com(cfg, writer, logger, model, loss_fn, trainloader, valloader, optimizer, scheduler, device)
        elif cfg['model']['arch'] == 'MIMOcom': # 
            trainer = Trainer_MIMOcom(cfg, writer, logger, model, loss_fn, trainloader, valloader, optimizer, scheduler, device)
        elif cfg['model']['arch'] == 'MIMOcomMultiWarp':
            trainer = Trainer_MIMOcomMultiWarp(cfg, writer, logger, model, loss_fn, trainloader, valloader, optimizer, scheduler, device)
        elif cfg['model']['arch'] == 'MIMOcomWho':
            trainer = Trainer_MIMOcomWho(cfg, writer, logger, model, loss_fn, trainloader, valloader, optimizer, scheduler, device)
        elif cfg['model']['arch'] == 'Single_agent':
            trainer = Trainer_Single_agent(cfg, writer, logger, model, loss_fn, trainloader, valloader, optimizer, scheduler, device)
        elif cfg['model']['arch'] == 'All_agents':
            trainer = Trainer_All_agents(cfg, writer, logger, model, loss_fn, trainloader, valloader, optimizer, scheduler, device)
        elif cfg['model']['arch'] == 'MIMO_All_agents':
            trainer = Trainer_MIMO_All_agents(cfg, writer, logger, model, loss_fn, trainloader, valloader, optimizer, scheduler, device)
        else: 
            raise ValueError('Unknown arch name for training')

        model_path = trainer.train()


        # ================ Val + Test ================

        te_loader = data_loader(
            data_path,
            split=cfg["data"]['test_split'],
            is_transform=True,
            img_size=(cfg["data"]["img_rows"], cfg["data"]["img_cols"]),
            target_view=cfg["data"]["target_view"],
            commun_label=if_commun_label)

        n_classes = te_loader.n_classes
        testloader = data.DataLoader(te_loader, batch_size=cfg["training"]["batch_size"], num_workers=8)

        # load best weight
        trainer.load_weight(model_path)
        _ = trainer.evaluate(testloader)



