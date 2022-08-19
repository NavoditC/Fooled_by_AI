
# suppress error
import logging as logging
import sys as sys
logging.disable(sys.maxsize)

# packages for md_sim
import gym
import math
import random
import numpy as np
import metadrive

# import the library
import torch
from PIL import Image
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import models, transforms, utils
from scratchai import * # comes with required packages, functions

# import internal modules
from PGD import pgd_linf, pgd_linf_targ

Image_Tensor_Class = transforms.ToTensor()                                      # creates Tensor class to change np.ndarray or PIL.Image to torch.Tensor

def initStats():
    global class_id_list_actual, class_id_list_adv, class_id_list_rp
    class_id_list_actual, class_id_list_adv, class_id_list_rp = [], [], []

def showStats():
    # convert to np
    list_actual = np.array(class_id_list_actual)
    list_adv = np.array(class_id_list_adv)
    list_rp = np.array(class_id_list_rp)

    # data analysis
    vals,counts = np.unique(list_actual, return_counts=True)
    print(vals[np.argmax(counts)], np.max(counts))
    print(np.sum(list_adv == 920)/list_adv.shape[0])
    # changed = list_adv[list_adv == 920][0] != list_rp[list_adv == 920][0]
    # print(np.sum(changed)/changed.shape[0])
    changed = list_rp[list_adv == 920] != 920
    if changed.shape[0] == 0:
        print(0.0)
    else:
        print(np.sum(changed)/changed.shape[0])

def showTensorImage(img_to_show):
    '''
    Takes the image of type torch.Tensor and displays it.
    img_to_show:    input image (torch.Tensor)

    Returns nothing.
    '''

    imgutils.imshow(img_to_show)

def show_plot(img_to_show, entry = 0):
    '''
    Function to plot/display images which are of type np.ndarray
    img_to_show:    input image to be displayed (torch.Tensor)
    entry:          unique id while saving the images (int)

    Returns nothing.
    '''
    if torch.is_tensor(img_to_show):
        img_to_show = img_to_show.permute(1, 2, 0)
        img_to_show = img_to_show.numpy()
        # showTensorImage(img_to_show)
    # else:
    # angle = -90 # degrees
    # img_to_show = ndimage.rotate(img_to_show, angle, reshape=True)
    plt.imshow(img_to_show, interpolation='nearest')
    plt.axis('off')
    plt.show()              # WORKS!
    # plt.savefig('Images/adv_img_' + str(entry) + '.png')

def randPad(image, to_size = (70, 70)):    # assuming image is torch.Tensor
    '''
    Function to implement Randomized Padding technique used for adversarial
    defense.
    image:      image to be padded, assumed to be square (torch.Tensor)
    to_size:    size of resized image (tuple)

    Resizes the input image to an arbitrary size (within defined range) and then
    randomly pads with 0 so as the resulting image is same size that of input image.

    Returns image of same shape.
    '''

    # set the seed before generating pseudo-random numbers
    random.seed(123)

    # resize the tensor by specified margin
    x = image.shape[-1] # size of original square image
    resize_margin = 20
    size_diff = random.randint(10, resize_margin)
    to_size = (x - size_diff, x - size_diff)
    transform = transforms.Resize(to_size)
    image = transform(image)    # tested successfully

    # randomly pad resized tensor to regain orginal size
    y = image.shape[-1]
    # calculate padding
    pad_left = random.randint(0, x - y)
    pad_right = size_diff - pad_left
    pad_top = random.randint(0, x - y)
    pad_bottom = size_diff - pad_top

    # make padding
    padding = (pad_left, pad_right, pad_top, pad_bottom)
    padded_image = F.pad(image, padding)    # tested successfully

    assert padded_image.shape[0] == image.shape[0] or padded_image.shape[1] == image.shape[1] or padded_image.shape[2] == image.shape[2] is True, "Padded image not same size"

    return padded_image

# function handle to get prediction more easily
def get_prediction(image, model):
    '''
    Function to predict the classification of the given image by the given NN.
    image:  input image to be classified (torch.Tensor)
    model:  neural network used for classification (torch.nn.model)

    Returns 3 results: class label, class index, confidence
    '''

    #assumes img and net are datasets and models trained using imagenet dataset
    confidences = model(image.unsqueeze(0))
    class_idx = torch.argmax(confidences, dim=1).item()
    class_label = datasets.labels.imagenet_labels[class_idx]
    return class_label, confidences[0, class_idx].item(), class_idx

def isTrafficLight(class_id):
    '''
    Function to check if given class index corresponds to traffic light class on
    the ImageNet dataset.
    class_id:   input class id (int)

    Returns true if traffic light.
    '''
    return class_id == 920

def makeSetup():
    '''
    Function to create setup environment (passed as a configuration dictionary)
     in MetaDrive.
    Returns nothing.
    '''

    # Setup road scenario
    global config_dict, HardAutoPilot
    HardAutoPilot = True
    config_dict = dict(use_render = False, manual_control = HardAutoPilot)               # initialise config dictionary, manual_control means keyboard control
    config_dict["map_config"] = {"type": "block_sequence", "config": "OCOC", "lane_num": 8}

    # Setup Vision-Based Observation
    config_dict["offscreen_render"] = True
    config_dict["vehicle_config"] = {"image_source": "rgb_camera"}
    config_dict["vehicle_config"]["rgb_camera"] = (100, 100)

def loadCNN():
    '''
    Function to load the CNN to be used for classification.

    Returns nothing.
    '''

    # get the CNN
    global net
    net = models.resnet18(pretrained=True).eval()                               # load resnet

def loadGraphics():
    global full_moon_img

    full_moon_img = imgutils.load_img('C:\\Users\\Hp\\Desktop\\input_images\\fair_moon.jpg')

    # resizing on PIL object
    full_moon_img = full_moon_img.resize((20, 20))  # WORKS!    # (20, 20) as max size works
    # resizing on PIL object

    full_moon_img = Image_Tensor_Class(full_moon_img)
    full_moon_img = full_moon_img.permute(1, 2, 0)
    full_moon_img = full_moon_img.numpy()


def addMoon(sample_image):
    # full_moon_img already loaded in get_example

    # first make sample image a np array
    if torch.is_tensor(sample_image):
        sample_image = sample_image.permute(1, 2, 0)
        sample_image = sample_image.numpy()

    # get pivot coordinates
    x = sample_image.shape[1]   # FYI x and y are both same as square image
    m = full_moon_img.shape[0]
    pivot_x = math.floor((x - m)/2)
    sky_height = 0.27*x

    pivot_y = math.floor(sky_height*0.5) - math.floor(m/2)

    # get black rectangle
    # black_rect = np.zeros((100, x, 3))
    black_rect = np.zeros((math.floor(sky_height), x, 3))

    # paste black rectangle
    sample_image[:math.floor(sky_height), :, :] = black_rect

    # paste the moon image
    sample_image[pivot_y:pivot_y + m, pivot_x:pivot_x + m, :] = full_moon_img

    return Image_Tensor_Class(sample_image)

def runDigitalTwin(showTime = False):
    '''
    Starts the simulation on MetaDrive.
    showTime:   flag to display time taken to execute key steps (boolean)

    Returns nothing.
    '''
    if showTime: import time

    # run the simulation for given timesteps
    # ======================= Initialise Env ======================== #
    env = gym.make("MetaDrive-validation-v0", config = config_dict)
    env.reset()
    class_idx = 0
    env.current_track_vehicle.expert_takeover = HardAutoPilot
    # =============================================================== #

    for i in range(1000):
        # take one step
        if isTrafficLight(class_idx):
            A1 = 0      # no steering
            A2 = -1     # dacc

            class_idx = 0                                                       # reset class_idx to move car after attack is over
        else:
            A1 = 0      # no steering
            A2 = 1      # acc

        obs, reward, done, info = env.step([A1, A2])

        # get image from camera
        img = obs['image']                                                      # img is NORMALISED (0-1) numpy.ndarray of size [100, 100, 3]
        angle = -90 # degrees                                                   # somehow img is rotated, FIXED
        img = ndimage.rotate(img, angle, reshape=True)

        # ========================= #
        # Integrate OpenCV code here
        img = addMoon(img)
        # ========================= #

        # convert np.ndarray to torch.Tensor
        # img = Image_Tensor_Class(img)                                           # img is torch.Tensor of size [3, 100, 100]

        if i > 40 and i < 141:

            # get initial prediction for precision targetting
            _, _, class_idx = get_prediction(img, net)
            class_id_list_actual.append(class_idx)
            if showTime: t2 = time.time()

            # increase one dimension
            img = img[None]                                                     # img is tensor of size [1, 3, 100, 100]

            # make targetted PGD attack
            delta = pgd_linf_targ(net, img, y = torch.tensor([class_idx]), epsilon=0.2, alpha=1e-2, num_iter=5, y_targ=920)
            if showTime: t3 = time.time()
            img = img + delta

            # get adversarial prediction
            _, _, class_idx = get_prediction(img[0], net)                       # img is tensor of size [1, 3, 100, 100]
            class_id_list_adv.append(class_idx)

            # ================================ #
            # Integrate Randomised Padding here
            img = randPad(img[0])
            _, _, class_idx = get_prediction(img, net)
            class_id_list_rp.append(class_idx)
            # ================================ #

            # print("M", class_idx)
            if showTime: print("{:.6f} s".format(t3 - t2))

        # env.render(text={"Auto-Drive (Switch mode: T)": "on" if env.current_track_vehicle.expert_takeover else "off",})
        if i == 141:
            return
        if done:
            # ======================= Re-initialise Env ====================== #
            env.reset()
            class_idx = 0
            # ================================================================ #

if __name__ == "__main__":
    '''
    Main function to kick start Digital Twin simulation.
    '''

    makeSetup()
    loadCNN()
    loadGraphics()
    initStats()
    runDigitalTwin()
    showStats()
