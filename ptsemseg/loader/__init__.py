import json

from ptsemseg.loader.airsim_loader import airsimLoader


def get_loader(name):
    """get_loader

    :param name:
    """
    return {
        "airsim": airsimLoader,
    }[name]

