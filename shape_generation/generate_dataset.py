import os
import shutil
from   datetime import datetime
import progress.bar
from random import choice

from shape_generation.generate_random_shape import generate

# Generate full dataset w/ parameters
LOW_PTS,MID_PTS,HIGH_PTS = 4, 12, 25
n_shapes       = 10
time           = datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
dataset_dir    = './data/dataset_'+time+'/'
shapes_dir = dataset_dir+'shapes/'
texts_dir = dataset_dir+'textures/'
colors_dir = dataset_dir+'colors/'
filename       = 'shape'
classes = ['low/', 'mid/', 'high/']
textures = ['/', '|', '-', '+', 'x', 'o', 'O', '.', '*']
colors = ['red', 'orange', 'yellow', 'green', 'blue', 'slategray', 'pink', 'cyan', 'magenta']

def generate_dataset():
    # Create directories if necessary
    if not os.path.exists(shapes_dir):
        for cls in classes:
            os.makedirs(shapes_dir+cls)
    if not os.path.exists(texts_dir):
        for cls in classes:
            os.makedirs(texts_dir+cls)
    if not os.path.exists(colors_dir):
        for cls in classes:
            os.makedirs(colors_dir+cls)

    # Generate dataset of just shapes
    bar = progress.bar.Bar('Generating shapes only', max=n_shapes * len(classes))
    for i in range(0,n_shapes):
        for cls in classes:

            # class-specific params
            if cls is 'low/':
                n_pts = n_sampling_pts =  LOW_PTS
            elif cls is 'mid/':
                n_pts = n_sampling_pts =  MID_PTS
            else:
                n_pts = n_sampling_pts =  HIGH_PTS

            generate(filename+'_'+str(i), n_pts, n_sampling_pts, plot_pts=False)
            img  = filename+'_'+str(i)+'.png'              
            shutil.move(img,  shapes_dir+cls)
            bar.next()

    # End bar
    bar.finish()

    # Generate dataset of shapes with random textures
    bar = progress.bar.Bar('Generating shapes with random textures', max=n_shapes * len(classes))
    for i in range(0,n_shapes):
        for cls in classes:

            # class-specific params
            if cls is 'low/':
                n_pts = n_sampling_pts =  LOW_PTS
            elif cls is 'mid/':
                n_pts = n_sampling_pts =  MID_PTS
            else:
                n_pts = n_sampling_pts =  HIGH_PTS

            text = choice(textures)
            generate(filename+'_'+str(i), n_pts, n_sampling_pts, hatch=text, plot_pts=False)
            img  = filename+'_'+str(i)+'.png'               
            shutil.move(img,  texts_dir+cls)
            bar.next()

    # End bar
    bar.finish()

    # Generate dataset of shapes with random textures and random colors
    bar = progress.bar.Bar('Generating shapes with random textures and random colors', max=n_shapes * len(classes))
    for i in range(0,n_shapes):
        for cls in classes:

            # class-specific params
            if cls is 'low/':
                n_pts = n_sampling_pts =  LOW_PTS
            elif cls is 'mid/':
                n_pts = n_sampling_pts =  MID_PTS
            else:
                n_pts = n_sampling_pts =  HIGH_PTS

            text = choice(textures)
            clr = choice(colors)
            generate(filename+'_'+str(i), n_pts, n_sampling_pts, hatch=text, fillColor=clr, plot_pts=False)
            img  = filename+'_'+str(i)+'.png'              
            shutil.move(img,  colors_dir+cls)
            bar.next()

    # End bar
    bar.finish()

    return shapes_dir, texts_dir, colors_dir