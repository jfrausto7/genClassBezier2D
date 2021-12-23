from shapes_utils import *

# constraints
plot_pts       = True
magnify        = 1.5
xmin           =-1.0
xmax           = 1.0
ymin           =-1.0
ymax           = 1.0

def generate(filename, n_pts, n_sampling_pts, radius=0.5, edgy=1.0, fillColor='red', hatch=''):
  # generate shape and write to csv
  shape = Shape(filename,
              None,
              n_pts,
              n_sampling_pts,
              [radius],
              [edgy],
              fillColor,
              hatch)
  shape.generate(magnify = magnify)
  shape.generate_image(plot_pts = plot_pts,
                      xmin     = xmin,
                      xmax     = xmax,
                      ymin     = ymin,
                      ymax     = ymax)
  shape.write_csv()

  return shape.name

# generate('shape', 5, 50)