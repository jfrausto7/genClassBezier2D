import os
import random
import shutil
from   datetime import datetime
from generate_random_shape import generate
import progress.bar

from shapes_utils import *
from generate_random_shape import generate

# Generate full dataset w/ parameters
n_sampling_pts = 5
n_shapes       = 50
time           = datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
dataset_dir    = 'dataset_'+time+'/'
img_dir        = dataset_dir+'images/'
csv_dir        = dataset_dir+'csv/'
filename       = 'shape'

# Create directories if necessary
if not os.path.exists(img_dir):
    os.makedirs(img_dir)
if not os.path.exists(csv_dir):
    os.makedirs(csv_dir)

# Generate dataset
bar = progress.bar.Bar('Generating shapes', max=n_shapes)
for i in range(0,n_shapes):

    generated = False
    while (not generated):
        n_pts  = random.randint(3, 7)
        # radius = np.random.uniform(0.0, 1.0, size=n_pts)
        # edgy   = np.random.uniform(0.0, 1.0, size=n_pts)

        generate(filename+'_'+str(i), n_pts, n_sampling_pts)
        img  = filename+'_'+str(i)+'.png'    
        csv = filename+'_'+str(i)+'.csv'              
        shutil.move(img,  img_dir)
        shutil.move(csv, csv_dir)
        generated = True

    bar.next()

# End bar
bar.finish()