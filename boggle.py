# suppress error
import logging as logging
import sys as sys
logging.disable(sys.maxsize)

# import the library
import math
import torch
import random
import numpy as np
import torchvision.utils
from PIL import Image
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import models, transforms
from scratchai import * # comes with required packages, functions

# import internal modules
from PGD import pgd_linf_targ

Image_Tensor_Class = transforms.ToTensor()                                      # creates Tensor class to change np.ndarray or PIL.Image to torch.Tensor

# target_class = 920 # imagenet id for traffic light
# label = ("traffic light, traffic signal, stoplight")

def prettyPrint(lab, ind, conf):
    '''
    Prints the prediction outcomes returned by the function get_prediction in
    visually pleasing format.
    lab:    label (string)
    ind:    class index (string)
    conf:   confidence for classification (string)

    Returns nothing.
    '''

    print(f"Prediction: {lab}")
    print(f"Class Index: {ind}")
    print(f"Confidence: {conf}")

def printSectionBreak(middle_part):
    '''
    Prints a section heading to cmd
    middle_part:    section title (string)

    Returns nothing.
    '''

    out_str = "\n# =================================== " + middle_part + " ============================================= #\n"
    print(out_str)

def show_plot(img_to_show, entry = 0):
    '''
    Function to plot/display images which are of type np.ndarray
    img_to_show:    input image to be displayed (torch.Tensor)
    entry:          unique id while saving the images (int)

    Returns nothing.
    '''
    if torch.is_tensor(img_to_show):
        # img_to_show = img_to_show.permute(1, 2, 0)
        # img_to_show = img_to_show.numpy()
        showTensorImage(img_to_show)
    else:
        plot = plt.imshow(img_to_show)
        plt.axis('off')
        plt.show()              # WORKS!


    # 1
    # img_to_show = img_to_show.permute(1, 2, 0)
    # img_to_show = img_to_show.numpy()
    #
    # plt.imshow(img_to_show, interpolation = "nearest")

    # 2
    # angle = -90 # degrees
    # img_to_show = ndimage.rotate(img_to_show, angle, reshape=True)
    # plt.axis('off')
    # plt.imshow(img_to_show, interpolation='nearest')
    # plt.savefig('Images/sim_img_' + str(entry) + '.png')

    # 3
    # transform = transforms.ToPILImage()
    # img_to_show = transform(img_to_show)
    # img_to_show.show()

def showTensorImage(img_to_show):
    '''
    Takes the image of type torch.Tensor and displays it.
    img_to_show:    input image (torch.Tensor)

    Returns nothing.
    '''

    imgutils.imshow(img_to_show)

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
    # random.seed(123)

    # resize the tensor by specified margin
    x = image.shape[-1] # size of orinal square image
    resize_margin = 50
    size_diff = random.randint(1, resize_margin)
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

def addMoon(sample_image):
    # full_moon_img already loaded in get_example

    # first make sample image a np array
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


def run_example(sample_image, CNN):
    '''
    Function to run a test example of adversarial attack. Prints results to cmd.
    sample_image:   test image (torch.Tensor)
    CNN:            NN used for classification

    Returns nothing.
    '''

    # sample_image = randPad(sample_image)

    # =================================== Initial Prediction ============================================= #
    printSectionBreak("Initial Prediction")

    # use the provided get_prediction function to predict the class of the moon image
    class_label, confidences, class_idx = get_prediction(sample_image, CNN)  # img is tensor of size [3, 224, 224]

    # print the results in nice manner
    prettyPrint(class_label, confidences, class_idx)

    # =================================== Add Moon ============================================= #
    sample_image = addMoon(sample_image)
    # =================================== PGD Attack ============================================= #

    # img is already a tensor
    sample_image = sample_image[None] # increases one dimension

    delta = pgd_linf_targ(CNN, sample_image, y = torch.tensor([class_idx]), epsilon=0.2, alpha=1e-1, num_iter=10, y_targ=920)
    sample_image = sample_image + delta
    # imgutils.imshow(sample_image) # works with tensors
    show_plot(sample_image)

    # =================================== Adversarial Prediction ============================================= #
    printSectionBreak("Adversarial Prediction")

    class_label, confidences, class_idx = get_prediction(sample_image[0], CNN)

    # print the results in nice manner
    prettyPrint(class_label, confidences, class_idx)

def get_example():
    '''
    Function to fetch sample image and run the adversarial attack algorithm on the
    same image.

    Returns nothing.
    '''
    global full_moon_img
    # full_moon_img = Image.open('C:\\Users\\Hp\\Desktop\\input_images\\yellow_full_moon.jpg')
    # full_moon_img = np.asarray(full_moon_img)   # converted into np array
    full_moon_img = imgutils.load_img('C:\\Users\\Hp\\Desktop\\input_images\\yellow_moon.jpg')

    # resizing on PIL object
    full_moon_img = full_moon_img.resize((15, 15))  # WORKS!
    # resizing on PIL object

    full_moon_img = Image_Tensor_Class(full_moon_img)
    full_moon_img = full_moon_img.permute(1, 2, 0)
    full_moon_img = full_moon_img.numpy()

    sample_image_path = 'C:\\Users\\Hp\\Desktop\\input_images\\sim_img_9.PNG'

    # load and preprocess the moon image
    # ex_image = Image.open(sample_image_path)
    # ex_image = np.asarray(ex_image)
    ex_image = imgutils.load_img(sample_image_path)

    # ex_image = imgutils.get_trf('rz256_cc224_tt_normimgnet')(ex_image) #normalize and reshape the input image
    ex_image = Image_Tensor_Class(ex_image)

    # load resnet
    resnet = models.resnet18(pretrained=True).eval()

    run_example(ex_image, resnet)

if __name__ == "__main__":
    '''
    Main function to kick start the pipeline.
    '''

    get_example()
