import os

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

def make_movie(filepath_figures, filepath_directory_movie, movie_name,
               framerate, start_number, num_figures):

    #=== Creating Directory for Movie ===#
    if not os.path.exists(filepath_directory_movie):
        os.makedirs(filepath_directory_movie)

    #=== Create Movie ===#
    padding_string = 'pad=ceil(iw/2)*2:ceil(ih/2)*2'
    ffmpeg_string = 'ffmpeg -y ' +\
                    '-r %d '%(framerate) +\
                    '-start_number %d '%(start_number) +\
                    '-pattern_type glob -i ' + '"' + filepath_figures + '_*.png" ' +\
                    '-vframes %d '%(num_figures) +\
                    '-pix_fmt yuv420p ' +\
                    '-vf "%s" '%(padding_string) +\
                    filepath_directory_movie + '/' + movie_name + '.mp4'
    os.system(ffmpeg_string)
