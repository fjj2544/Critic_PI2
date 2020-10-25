# -*- coding: UTF-8 -*-

import os
import imageio
import re

def create_gif(image_list, gif_name):

    frames = []
    for image_name in image_list:
        if image_name.endswith('.png'):
            print(image_name)
            frames.append(imageio.imread(image_name))
    # Save them as frames into a gif
    imageio.mimsave(gif_name, frames, 'GIF', duration = 0.1)

    return

def takeSecond(elem):
    return elem[1]
def main():

    path=r'./reward/'
    image_list=[ path+img for img in  os.listdir(path)]
    number_list = [int(re.findall(r'\d+', ss)[0]) for ss in image_list]
    image_list = [(a,b) for a,b in zip(image_list,number_list)]
    image_list.sort(key=takeSecond)
    image_list = [item[0] for item in image_list]
    gif_name = 'reward.gif'
    create_gif(image_list, gif_name)

if __name__ == "__main__":
    main()