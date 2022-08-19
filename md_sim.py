
# import necessary packages
import gym
import numpy as np
import metadrive
from matplotlib import pyplot as plt
import scipy.ndimage as ndimage

def show_plot(img_to_show, entry):

    angle = -90 # degrees
    img_to_show = ndimage.rotate(img_to_show, angle, reshape=True)
    plt.axis('off')
    plt.imshow(img_to_show, interpolation='nearest')
    # plt.savefig('Images/sim_img_' + str(entry) + '.png')

# Setup road scenario
config_dict =  dict(use_render = True,
                    manual_control = True)  #initialise config dictionary
config_dict["map_config"] = {"type": "block_sequence", "config": "OCOC", "lane_num": 3}

# Setup Vision-Based Observation
config_dict["offscreen_render"] = True
config_dict["vehicle_config"] = {"image_source": "rgb_camera"}
# config_dict["vehicle_config"]["image_source"] = "rgb_camera"
config_dict["vehicle_config"]["rgb_camera"] = (100, 100)

net = models.resnet18(pretrained=True).eval()  # load resnet

# initialise the env
env = gym.make("MetaDrive-validation-v0", config = config_dict)
env.reset()

for i in range(1000):
    obs, reward, done, info = env.step([0, 0])
    img = obs['image']
    show_plot(img, i)
    env.render(text={"Auto-Drive (Switch mode: T)": "on" if env.current_track_vehicle.expert_takeover else "off",})
    if done:
        env.reset()
