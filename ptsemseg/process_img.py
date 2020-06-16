import numpy as np
import scipy.misc as m
import time
import cv2
from torch.autograd import Variable
def generate_noise(img,noisy_type=None):
    '''
    Input: img: RGB image, noisy_type: string of noisy type
    generate noisy image
    * image must be RGB 
    '''
    image_batch = img.shape[0]
    img_ch  = img.shape[1]
    img_row = img.shape[2]
    img_col = img.shape[3]

    if noisy_type == 'occlusion':
        #print('Noisy_type: Occlusion')
        img[:,:,int(img_row/5):(img_row),:] = 0
    elif noisy_type == 'random_noisy':
        noise = Variable(img.data.new(img.size()).normal_(0, 0.8))
        img = img + noise
        img_np = img.data.numpy()
        #m.imsave('noisy_image.png',img_np[0].transpose(1,2,0))
    elif noisy_type == 'grayscale':
        #print('Noisy_type: Grayscale')
        img = np.dot(img[...,:3], [0.299, 0.587, 0.114])
    elif noisy_type == 'low_resolution':
        #print('Noisy_type: Low resolution (but now is original image)')
        pass
    else:
        # print('Noisy_type: original image)')
        pass

    return img

def recursive_glob(rootdir=".", suffix=""):
    """Performs recursive glob with given suffix and rootdir
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [
        os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames
        if filename.endswith(suffix)
    ]


def alpha_blend(input_image, segmentation_mask, alpha=0.5):
    """Alpha Blending utility to overlay RGB masks on RBG images
        :param input_image is a np.ndarray with 3 channels
        :param segmentation_mask is a np.ndarray with 3 channels
        :param alpha is a float value
    """
    blended = np.zeros(input_image.size, dtype=np.float32)
    blended = input_image * alpha + segmentation_mask * (1 - alpha)
    return blended


def convert_state_dict(state_dict):
    """Converts a state dict saved from a dataParallel module to normal
       module state_dict inplace
       :param state_dict is the loaded DataParallel model_state
    """
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    return new_state_dict


def get_logger(logdir):
    logger = logging.getLogger("ptsemseg")
    ts = str(datetime.datetime.now()).split(".")[0].replace(" ", "_")
    ts = ts.replace(":", "_").replace("-", "_")
    file_path = os.path.join(logdir, "run_{}.log".format(ts))
    hdlr = logging.FileHandler(file_path)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)
    return logger
