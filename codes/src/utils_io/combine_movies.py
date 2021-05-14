import os

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

def combine_movies(filepath_movie_1, filepath_movie_2,
                   filepath_directory_movie,
                   movie_name):
    ffmpeg_string = 'ffmpeg -y ' +\
                    '-i %s.mp4 '%(filepath_movie_1) +\
                    '-i %s.mp4 '%(filepath_movie_2) +\
                    '-filter_complex "[0:v]pad=iw*2:ih[int];[int][1:v]overlay=W/2:0[vid]" ' +\
                    '-map "[vid]" ' +\
                    '-c:v libx264 ' +\
                    '-crf 23 ' +\
                    '-preset veryfast ' +\
                    filepath_directory_movie + '/' + movie_name + '.mp4'
    os.system(ffmpeg_string)
