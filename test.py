import yaml
import torch
import argparse
import timeit
import numpy as np
import cv2
import os

from torch.utils import data
from ptsemseg.models import get_model
from ptsemseg.loader import get_loader
from ptsemseg.metrics import runningScore
from ptsemseg.utils import convert_state_dict
from ptsemseg.visual import draw_bounding
from ptsemseg.trainer import *

torch.backends.cudnn.benchmark = True


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


    parser.add_argument(
        "--model_path",
        nargs="?",
        type=str,
        default="Single_Agent.pkl",
        help="Path to the saved model",
    )


    args = parser.parse_args()

    # Set the gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    run_times = args.run_time

    with open(args.config) as fp:
        cfg = yaml.load(fp)

    # ============= Testing =============

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

    # test data loadeer 
    te_loader = data_loader(
        data_path,
        split=cfg["data"]['test_split'],
        is_transform=True,
        img_size=(cfg["data"]["img_rows"], cfg["data"]["img_cols"]),
        target_view=cfg["data"]["target_view"],
        commun_label=if_commun_label)

    testloader = data.DataLoader(te_loader, batch_size=cfg["training"]["batch_size"], num_workers=8)


    # Setup Model
    model = get_model(cfg, te_loader.n_classes).to(device) 

    # set up the model 
    if cfg['model']['arch'] == 'LearnWhen2Com': # Our when2com
        trainer = Trainer_LearnWhen2Com(cfg, None, None, model, None, None, None, None, None, device)
    elif cfg['model']['arch'] == 'LearnWho2Com': # Our who2com
        trainer = Trainer_LearnWho2Com(cfg, None, None, model, None, None, None, None, None, device)
    elif cfg['model']['arch'] == 'MIMOcom': # 
        trainer = Trainer_MIMOcom(cfg, None, None, model, None, None, None, None, None, device)
    elif cfg['model']['arch'] == 'MIMOcomMultiWarp':
        trainer = Trainer_MIMOcomMultiWarp(cfg, None, None, None, None, None, None, None, None, device)
    elif cfg['model']['arch'] == 'MIMOcomWho':
        trainer = Trainer_MIMOcomWho(cfg, None, None, model, None, None, None, None, None, device)
    elif cfg['model']['arch'] == 'Single_agent':
        trainer = Trainer_Single_agent(cfg, None, None, model, None, None, None, None, None, device)
    elif cfg['model']['arch'] == 'All_agents':
        trainer = Trainer_All_agents(cfg, None, None, model, None, None, None, None, None, device)
    elif cfg['model']['arch'] == 'MIMO_All_agents':
        trainer = Trainer_MIMO_All_agents(cfg, None, None, model, None, None, None, None, None, device)
    else: 
        raise ValueError('Unknown arch name for testing')


    print(args.model_path)
    # load best weight
    trainer.load_weight(args.model_path)

    # if you would like to obtain qual results or other stats, just change the output
    _ = trainer.evaluate(testloader)






