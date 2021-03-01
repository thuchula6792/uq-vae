import sys
import os

from paraview.simple import *

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

def plot_paraview(filepath_pvd, filepath_figure, cbar_RGB):
    #### disable automatic camera reset on 'Show'
    paraview.simple._DisableFirstRenderCameraReset()

    # create a new 'PVD Reader'
    data = PVDReader(FileName=filepath_pvd)

    # get active view
    renderView1 = GetActiveViewOrCreate('RenderView')
    renderView1.OrientationAxesVisibility = 0

    # show data in view
    data_pvdDisplay = Show(data, renderView1)

    #=== Colourbar ===#
    f_10LUT = GetColorTransferFunction('f_10')
    f_10LUT.RGBPoints = cbar_RGB
    f_10LUT.ScalarRangeInitialized = 1.0
    f_10LUTColorBar = GetScalarBar(f_10LUT, renderView1)
    f_10LUTColorBar.Title = ''
    f_10LUTColorBar.ComponentTitle = ''
    f_10LUTColorBar.WindowLocation = 'AnyLocation'
    f_10LUTColorBar.Position = [0.8014950166112955, 0.23803827751196172]
    f_10LUTColorBar.ScalarBarLength = 0.5393301435406705
    f_10LUTColorBar.ScalarBarThickness = 5
    f_10LUTColorBar.LabelFontSize = 5
    f_10LUTColorBar.LabelColor = [0.0, 0.0, 0.0]

    # current camera placement for renderView1
    renderView1.CameraPosition = [4.5215554222188254, 1.781219310617244, 4.6008047146248785]
    renderView1.CameraViewUp = [-0.19139311005181278, 0.9639050186425031, -0.18508320415556503]
    renderView1.CameraParallelScale = 1.7320508075688772

    # save screenshot
    SaveScreenshot(filepath_figure, renderView1, ImageResolution=[1432, 793], TransparentBackground=1)

    # reset
    paraview.simple.ResetSession()
