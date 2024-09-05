import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import yaml
import os
import argparse
from matplotlib.widgets import Button
from datetime import datetime
import sys
import cv2

from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import utils


def plot2D(cfg, data, custom_axis = [- 2, 2, - 2, 2], figsize=10, folder_out = None):
    fig, ax = plt.subplots(figsize=(figsize, figsize))  # Cambia los valores según necesites
    x = data[:, 0]
    y = data[:, 1]

    # Dibujar puntos
    ax.scatter(x, 1 - y, c='red')

    # Dibujar líneas
    for connection in cfg['CONNECTIONS']:
        start_point = connection[0]
        end_point = connection[1]
        x_values = [x[start_point], x[end_point]]
        y_values = [1 - y[start_point], 1 - y[end_point]]
        ax.plot(x_values, y_values, 'bo-', linewidth=2)  # 'bo-' significa puntos azules con líneas

    ax.axis(custom_axis)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Keypoints with Connections')
    if folder_out:
        plt.savefig(os.path.join(folder_out, str(datetime.now()) + '.png'))
        plt.close("all")
    else:
        plt.show()
        print('Plot2D is being shown. Make sure to close it afterwards. You can use plt.close(\'all\').')


def plot3D(cfg, data, folder_out = None, azimut=75):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax.axis('off')
    ax.set_facecolor('white')
    ax.set_box_aspect([1,1,1])  # Aspecto 1:1:1

    # plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)

    ax.scatter3D(-data[:, 0], data[:, 2], -data[:, 1], linewidths=1, marker='.', s=22, c='#1d61a0', depthshade=False)
    ax.elev = 0
    ax.azim = azimut

    for connection in cfg['CONNECTIONS']:
        if ((data[connection[0]][0] != 0) and (data[connection[1]][0] != 0)):
            ax.plot(-data[connection[:2], 0], data[connection[:2], 2], -data[connection[:2], 1], linewidth=0.8)

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
        
    if folder_out:
        plt.savefig(os.path.join(folder_out, str(datetime.now()) + '.png'))
        plt.close("all")
    else:
        plt.show()
        print('Plot3D is being shown. Make sure to close it afterwards. You can use plt.close(\'all\').')

def plot3D2(self, data, folder_out = None, custom_limits = [], figsize=10, azimut=75):
    fig, ax = plt.subplots(figsize=(figsize, figsize)) 
    ax = fig.add_subplot(111, projection='3d')
    # ax.axis('off')
    ax.set_facecolor('white')
    ax.set_box_aspect([1,1,1])  # Aspecto 1:1:1

    # plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)

    ax.scatter3D(-data[:, 0], data[:, 2], -data[:, 1], linewidths=1, marker='.', s=22, c='#1d61a0', depthshade=False)
    ax.elev = 0
    ax.azim = azimut

    for connection in cfg['CONNECTIONS']:
        if ((data[connection[0]][0] != 0) and (data[connection[1]][0] != 0)):
            ax.plot(-data[connection[:2], 0], data[connection[:2], 2], -data[connection[:2], 1], linewidth=0.8)

    if custom_limits != [] :
        ax.set_xlim([custom_limits[0], custom_limits[1]])
        ax.set_ylim([custom_limits[0], custom_limits[1]])
        ax.set_zlim([custom_limits[0], custom_limits[1]])

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
        
    if folder_out:
        plt.savefig(os.path.join(folder_out, str(datetime.now()) + '.png'))
    else:
        plt.show()

def plot3D_World(self, data, folder_out = None, azimut=-75):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.axis('off')
    ax.set_facecolor('white')
 

    max_range = np.array([data[:, 0].max()-data[:, 0].min(),
                            (data[:, 1].max()-data[:, 1].min()),
                            data[:, 2].max()-data[:, 2].min()]).max() / 2.0

    mid_x = (data[:, 0].max()+data[:, 0].min()) * 0.5
    mid_y = ((data[:, 1].max()+data[:, 1].min()) * 0.5 )
    mid_z = ((data[:, 2].max()+data[:, 2].min()) * 0.5 + 0.5)  # Posicionar

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.set_box_aspect([1,1,1])  # Aspecto 1:1:1
    ax.elev = 0
    ax.azim = azimut

    plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)

    ax.scatter3D(-data[:, 0], data[:, 2], -data[:, 1], linewidths=1, marker='.', s=22, c='#1d61a0', depthshade=False)

    for connection in cfg['CONNECTIONS']:
        if ((data[connection[0]][0] != 0) and (data[connection[1]][0] != 0)):
            ax.plot(-data[connection[:2], 0], data[connection[:2], 2], -data[connection[:2], 1], linewidth=0.8)

    plt.xlabel('X')
    plt.ylabel('Y')
    if folder_out:
        plt.savefig(os.path.join(folder_out, str(datetime.now()) + '.png'))
        plt.close("all")
    else:
        plt.show()
        print('Plot3D_world is being shown. Make sure to close it afterwards. You can use plt.close(\'all\').')


if __name__ == '__main__':
    # Load the configuration file
    old_cwd = os.getcwd()                                   # Needed if debugging is being done and cwd is not this file path. If not needed, comment this line
    os.chdir(os.path.dirname(os.path.abspath(__file__)))    # Needed if debugging is being done and cwd is not this file path. If not needed, comment this line
    config_path = '../config.yaml'
    with open(config_path, "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
        print(cfg)

    # Load the keypoints data and saving folder
    data = np.load('../../out_keypoints/ania_cerca.npy') #data = np.load('../../DATA/KEYPOINTS/POSE_HANDS_IMAGE_05/ania_cerca.npy')
    folder_out = '../../out_images'                         # ATTENTION: with re-runs the images generated won't be overwritten. They will be saved together with the previous ones since their names are based on the current time.
    if not os.path.exists(folder_out):
        os.makedirs(folder_out)
    print(data.shape)

    min_val_x = np.min(-data[:, :, 0])
    max_val_x = np.max(-data[:, :, 0])
    min_val_y = np.min(-data[:, :, 1])
    max_val_y = np.max(-data[:, :, 1])
    min_val_z = np.min(data[:, :, 2])
    max_val_z = np.max(data[:, :, 2])

    min_val = min(min_val_x, min_val_y, min_val_z)
    max_val = max(max_val_x, max_val_y, max_val_z)

    FOLDER_IMGS_OUT = '../../DATA/IMGS/ania_lejos/LATERAL'
    utils.create_folder(FOLDER_IMGS_OUT, reset=True, auto=True)
   
    for data_idx in data:
        plot3D2(cfg, data_idx, custom_limits=[min_val, max_val], folder_out=FOLDER_IMGS_OUT, azimut= 0)
    # plot3D(cfg, data[1])

    

    # Plot the 3D keypoints
    for i in range(data.shape[0]):
        plot3D(cfg, data[i], folder_out) # plot3D(cfg, data[1])

    # Generate a video from the image frames
    if True:                                                # Togle this to True if you want to generate a video from the images
        # Get the video parameters.
        filename = 'ania_cerca_keypoints'
        path_out_video = os.path.join(folder_out, filename+'.mp4')
        frame_rate = 30        
        img_files = os.listdir(folder_out)
        if img_files: 
            # The first image is used to get the resolution.
            img = cv2.imread(os.path.join(folder_out, img_files[0]))
            height, width, _ = img.shape
        else:
            print('No images found in the folder {}'.format(folder_out))
            exit()

        # Create the video writer object
        out = cv2.VideoWriter(path_out_video, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (width, height))
        print('Generating video called {} with frame rate {} and resolution {}x{}'.format(filename, frame_rate, width, height))

        # Write the images to the video
        img_files = sorted(img_files)
        for img_file in img_files:
            img = cv2.imread(os.path.join(folder_out,img_file))
            out.write(img)
        out.release()
    
    os.chdir(old_cwd)                                       # Needed if debugging is being done and cwd is not this file path. If not needed, comment this line
