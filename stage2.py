#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 09:32:44 2019

@author: wilfried
"""

from sys import exit

from astropy.table import Table
from astropy.io.votable import is_votable, writeto

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle

from copy import copy

def is_VOtable(fullname):
    """
    Check whether a file is a VOtable.
    
    Input
    ------
    fullname : string
        path+name of the file to test
    
    Returns True if it is a VOtable. False otherwise.
    """
    tag = is_votable(fullname)
    print("The file", fullname, "is a VOtable, right ?", tag)
    return tag

def write_array_to_vot(array, outputFile, isTable=False):
    """
    Writes an array or an astropy table into a .vot file.
    
    Input
    -----
    array : numpy array, astropy table
        The array to write into the file
    outputFile : string
        The file to write the array into
    isTable : boolean
        Whether the array is an astropy table or not.
    """
    
    #If it is an array it creates an astropy table
    if not isTable:
        array = Table(data=array)
        
    writeto(array, outputFile)
    return

def linear_fit(x, A, offset):
    """
    Compute a linear relation A*x+offset.
    
    Input
    -----
    x : numpy array
        input data
    A : float
        Slope coefficient
    offset : float
        x=0 Y-coordinate
        
    Returns a numpy array A*x+offset.
    """
    return A*x+offset

def convertCoords(coordinates, inSize=(200.0, 200.0), outSize=(31.0, 31.0), conversionFactor=1.0):
    '''
    Transforms the coordinates of a/many point(s) from one image to another
    
    Input
    -----
    coordinates : dictionnary or list of dictionnaries
        the coordinates of the points to convert form one image to another
    conversionFactor : float
        a numerical factor to convert the position from pixel to another relavant unit
    inSize : tuple/list
        the size of the image the points are from
    outSize : tuple/list
        the size of the image whereto we want to convert the positions of the points
        
    Returns a list of dictionnaries with transformed coordinates.
    '''
    
    try:
        np.shape(coordinates)[0]
    except:
        coordinates = [coordinates]
        
    for num, points in enumerate(coordinates):
        for pos, key in enumerate(points.keys()):
            coordinates[num][key] *= outSize[pos]/inSize[pos]*conversionFactor
    return coordinates

def computeGroupFWHM(wavelength, groups, verbose=True, model='Moffat'):
    '''
    Computes the FWHM at a given observed wavelength assuming a linearly decreasing relation for the FWHM with wavelength (calibrated on OII and OIII measurements at different redshifts) stars measurements for each group in the COSMOS field.
    
    Input
    -----
    groups : string or list of strings
        the group for each desired wavelength
    model : string
        the model to use, either Moffat or Gaussian
    verbose : boolean
        whether to print a message on screen with the computed FWHM or not
    wavelength : integer
        the wavelength(s) at which we want to compute the FWHM (must be in Angstroms)
    
    Returns a list of tuples with the group and the computed FWHM.
    '''
    
    #structure is as folows : number of the group, o2 FWHM, o3hb FWHM, mean redshift of the group
    if model == 'Moffat':
        listGroups = {'23' : [3.97, 3.29, 0.850458], '26' : [3.16, 2.9, 0.439973], '28' : [3.18, 3.13, 0.950289],
                      '32-M1' : [2.46, 1.9, 0.753319], '32-M2' : [2.52, 2.31, 0.753319], '32-M3' : [2.625, 2.465, 0.753319],
                      '51' : [3.425, 2.95, 0.386245], '61' : [3.2, 3.02, 0.364009], '79' : [2.895, 2.285, 0.780482], 
                      '84-N' : [2.49, 2.21, 0.727755], '30_d' : [2.995, 2.68, 0.809828], '30_bs' : [2.745, 2.45, 0.809828],
                      '84' : [2.835, 2.715, 0.731648], '34_d' : [2.89, 2.695, 0.857549], '34_bs' : [np.nan, np.nan, 0.85754],
                      '114' : [3.115, 2.81, 0.598849]}
    elif model == "Gaussian":
        listGroups = {'23' : [4.28, 3.65, 0.850458], '26' : [3.68, 3.34, 0.439973], '28' : [3.62, 3.26, 0.950289],
                      '32-M1' : [2.975,	2.58,  0.753319], '32-M2' : [3.16,	2.54, 0.753319], '32-M3' : [3.61,	3.3, 0.753319],
                      '51' : [3.75, 3.28, 0.386245], '61' : [3.915,	3.34, 0.364009], '79' : [3.29,	2.695, 0.780482],
                      '84-N' : [2.89,	2.58, 0.727755], '30_d' : [3.485,	3.11, 0.809828], '30_bs' : [3.185,	2.815, 0.809828],
                      '84' : [3.24,	3.055, 0.731648], '34_d' : [3.31,	2.995, 0.857549], '34_bs' : [3.3,	3.003, 0.85754],
                      '114' : [3.705,	3.315, 0.598849]}
    else:
        raise Exception("Model %s not recognised. Available values are %s" %(model, ["Moffat", "Gaussian"]))
    
    #lines wavelengths in Anstrom
    OIIlambda   = 3729 
    OIIIlambda  = 5007
    deltaLambda = OIIIlambda - OIIlambda
    
    try:
        np.shape(wavelength)[0]
    except:
        wavelength = [wavelength]
    try:
        np.shape(groups)[0]
    except:
        groups = [groups]
        
    #check wavelength and groups have the same size
    if len(wavelength) != len(groups):
        exit("Wavelength and group lists do not have the same length. Please provide exactly one group for each wavelength you want to compute.")
    
    #checking given group names exist
    for pos, name in enumerate(groups):
        name        = str(name)
        groups[pos] = name
        
        try:
            listGroups[name]
        except KeyError:
            exit("Given group %s is not correct. Possible values are %s" %(name, listGroups.keys()))
            
    outputList = []
    for wv, gr in zip(wavelength, groups):
        #lines wavelength are rest-frame wavelengths, but FWHM measurements were made at a certain redshift
        #A factor of (1+z) must be applied to deltaLambda and OII lambda
        grVals = listGroups[gr]
        slope  = (grVals[1] - grVals[0])/(deltaLambda*(1+grVals[2]))
        offset = grVals[0] - slope*OIIlambda*(1+grVals[2])
        
        FWHM = slope*wv+offset
        outputList.append((gr, FWHM))
        
        if verbose:
            print("FWHM at wavelength", wv, "angstroms in group", gr, "is", FWHM)
            
    return outputList
        
    
    
    

def printSimpleStat(catalog, unit=None):
    """
    Print basic stats such as median and mean values, as well as 1st and 3rd quantiles.
    
    Input
    -----
    catalog : array/astropy table/list or list of arrays/astropy tables/lists
        array from which the statistic is computed
    unit: astropy unit
        unit of the array if there is one
    """

    try:
        np.shape(catalog[1])
    except IndexError:
        catalog = [catalog]
    
    for cat, num in zip(catalog, range(len(catalog))):
        if unit is not None:
            cat = cat*unit
            
        print("Stat for catalog number", num, ":")
        print("Maximum separation is", str(np.max(cat)) + ".")
        print("Mean separation is", str(np.mean(cat)) + ".")
        print("Median separation is", str(np.median(cat)) + ".")
        print("1st quantile is", str(np.quantile(cat, 0.25)) + ".")
        print("3rd quantile is", str(np.quantile(cat, 0.75)) + ".")
        
    return      

def uniqueArr(tables, arraysToBeUnique):
    """
    Apply a mask from np.unique on arraysToBeUnique for many arrays.
    
    Input
    -----
    tables : table/array or list of tables/arrays
        tables to which the mask is applied
    arraysToBeUnique : table/array or list of tables/arrays
        tables or arrays from which the mask is computed (with np.unique)
        
    Returns tables with the mask applied.
    """
    
    #Transform into a list if it is an array
    try:
        np.shape(tables[1])
    except IndexError:
        tables = [tables]
    try:
        np.shape(arraysToBeUnique[1])
    except IndexError:
        arraysToBeUnique = [arraysToBeUnique]
        
    for num, uniq in zip(range(len(tables)), arraysToBeUnique):    
        arr, indices = np.unique(uniq, return_index=True)
        tables[num]  = tables[num][indices]
        
    return tables

def maskToRemoveVal(listOfArrays, val=None, keep=True, astroTableMask=False):
    """
    Computes a mask by finding occurences in a list of arrays.
    
    Input
    -----
    listOfArrays : list of numpy arrays
        the list of arrays from which the mask is built
    val : float or None
        the value to find. If val=None, it looks for nan values.
    keep : boolean
        if True, it builds a mask with True everywhere the value val is encountered. If False, it does the opposite
    astroTableMask : boolean
        if True returns a mask from the astropy table column instead of looking for some value/nans with False values everywhere the data is masked
    
    Returns a mask as a numpy array.
    """
    
    shp = listOfArrays[0].shape
    #Checking that arrays have the same shape
    for array in listOfArrays[1:]:
        if shp != array.shape:
            exit("Arrays do not have the same dimensions, thus making the masking operation unfit. Exiting.")
  
    #Constructing first mask
    if astroTableMask:
        tmp = np.logical_not(listOfArrays[0].mask)
    elif val is None:
        tmp = np.logical_not(np.isnan(listOfArrays[0]))
    else:
        tmp = listOfArrays[0] == val
        if not keep:
            tmp = np.logical_not(tmp)
        
    #Applying logical and on all the masks
    for (num, array) in enumerate(listOfArrays[1:]):
        #consider we are looking for nan in the arrays
        if astroTableMask:
            tmp = np.logical_and(tmp, np.logical_not(array.mask))  
        elif val is None:
            tmp = np.logical_and(tmp, np.logical_not(np.isnan(array)))
        else:
            if keep:
                tmp = np.logical_and(tmp, array==val)
            else:
                tmp = np.logical_and(tmp, array != val)
    return tmp

def applyMask(listOfArrays, mask):
    """
    Apply the same mask to a list of arrays and return the new arrays.
    
    Input
    -----
    listOfArrays : list of numpy arrays
        the list of arrays the mask is applied to
    mask : numpy array
        the mask to apply
        
    Returns the list of arrays with the mask applied. If len(listOfArrays) is 1, it returns only an array instead of a list of arrays with one object.
    """

    for (num, array) in enumerate(listOfArrays):
        if len(listOfArrays) == 1:
            listOfArrays = array[mask]
        else:
            listOfArrays[num] = array[mask]
    return listOfArrays

def findWhereIsValue(listOfArrays, val=None):
    """
    Find and print the first position where a value is found within a list of arrays.
    
    Input
    -----
    listOfArrays : list of numpy arrays
        list from which the value val is searched
    val : float or None
        value to look for. If val=None, it looks for nan values.
        
    Returns a list of booleans with the same length as listOfArrays, with True when the value was found in the array and False otherwise.
    """
    
    returnArr = []
    
    for (num, array) in enumerate(listOfArrays):
        if val is None:
            if np.any(np.isnan(array)):
                returnArr.append(True)
                print("A nan was found at position", np.where(np.isnan(array))[0], "within array number", num)
            else:
                returnArr.append(False)
                print("No nan was found in array number", num)
        else:
            if np.asarray(np.where(array==val)).shape[1] == 0:
                returnArr.append(False)
                print("No value", val, "found within array number", num)
            else:
                returnArr.append(True)
                print("Value", val, "found at position", np.where((array==val))[0], "within array number", num)
    return returnArr
                
def checkDupplicates(master, names=None):
    """
    Check if galaxies are found multiple times in an array by looking for duplicates of (RA, DEC) pairs.
    
    Input
    -----
    master : list of structured numpy arrays (with 'RA' and 'DEC' fields)
        a list of structured arrays to check
    names : list of strings
        the names of the arrays
    """
    
    if (names is None) or (len(names) != len(master)):
        try:
            len(names) != len(master)
            print("Given names were not enough. Using position in the list as name instead.")
        except TypeError:
            pass
        names = np.char.array(['catalog nb ']*len(master)) + np.char.array(np.array(range(len(master)), dtype='str'))
    
    for catalog, nameCat in zip(master, names):
        cnt = True
        for ra, dec, nb in zip(catalog['RA'], catalog['DEC'], range(catalog['RA'].shape[0])):
            
            where1 = np.where(catalog['RA']==ra)[0]
            where2 = np.where(catalog['DEC']==dec)[0]
            
            if (len(where1)>1) and (len(where2)>1):
                
                flag = True
                for w in where2:
                    
                    if flag and (w in where1):
                        print("RA =", ra, "deg and DEC =", dec, "deg galaxy (line " + str(nb) + ") is present more than once in catalog", nameCat)
                        flag = False
                        cnt  = False
        if cnt:
            print("All the galaxies are only listed once in the catalog", nameCat)     
    return


def asManyHists(numPlot, data, bins=None, weights=None, hideXlabel=False, hideYlabel=False, hideYticks=False, hideXticks=False,
                placeYaxisOnRight=False, xlabel="", ylabel='', color='black',
                label='', zorder=0, textsize=24, showLegend=False, legendTextSize=24,
                xlim=[None, None], locLegend='best', tickSize=24, title='', titlesize=24,
                outputName=None, overwrite=False, tightLayout=True, integralIsOne=None,
                align='mid', histtype='stepfilled', alpha=1.0, cumulative=False, legendNcols=1, hatch=None, orientation='vertical'):

    """
    Function which plots on a highly configurable subplot grid 1D histograms. A list of data can be given to have multiple histograms on the same subplot.

    Input
    -----
    align : 'left', 'mid' or 'right'
        how to align bars respective to their value
    alpha : float
        how transparent the bars are
    bins : int or list of int
        if an integer, the number of bins. If it is a list, edges of the bins must be given.
    color : list of strings/chars/RGBs
        color for the data. It can either be a string, char or RGB value.
    cumulative : boolean
        whether to plot the cumulative distribution (where each bin equals the sum of the values in the previous bins up to this one) or the histogram
    data: numpy array, list of numpy arrays
        the data
    hatch : char
        the hatching pattern
    hideXlabel : boolean
        whether to hide the x label or not
    hideXticks : boolean
        whether to hide the x ticks or not
    hideYlabel : boolean
        whether to hide the y label or not
    hideYticks : boolean
        whether to hide the y ticks or not
    histtype : 'bar', 'barstacked', 'step', 'stepfilled'
        how the histogram is plotted. Bar puts histograms next to each other. Barstacked stacks them. Step plots unfilled histograms. Stepfilled generates a filled histogram by default.
    integralIsOne : boolean or list of boolean
        whether to normalize the integral of the histogram
    label : string
        legend label for the data
    legendNcols : int
        number of columns in the legend
    legendTextSize : int
        size for the legend
    locLegend : string, int
        position where to place the legend
    numPlot : int (3 digits)
        the subplot number
    orientation : str
        orientation of the bars
    outputName : str
        name of the file to save the graph into. If None, the plot is not saved into a file
    overwrite : boolean
        whether to overwrite the ouput file or not
    placeYaxisOnRight : boolean
        whether to place the y axis of the plot on the right or not
    textsize : int
        size for the labels
    showLegend : boolean
        whether to show the legend or not
    tickSize : int
        size of the ticks on both axes
    tightLayout : boolean
        whether to use bbox_inches='tight' if tightLayout is True or bbox_inches=None otherwise
    weights : numpy array of floats or list of numpy arrays
        the weights to apply to each value in data
    xlabel : string
        the x label
    xlim : list of floats/None
        the x-axis limits to use. If None is specified as lower/upper/both limit(s), the minimum/maximum/both values are used
    ylabel : string
        the y label
    ylim : list of floats/None
        the y-axis limits to use. If None is specified as lower/upper/both limit(s), the minimum/maximum/both values are used
    zorder : int, list of ints for many plots
        whether the data will be plot in first position or in last. The lower the value, the earlier it will be plotted
        
    Return current axis, hist values and bins.
    """
    
    ax1 = plt.subplot(numPlot)
    ax1.yaxis.set_ticks_position('both')
    ax1.xaxis.set_ticks_position('both')
    ax1.set_title(title, size=titlesize)
    ax1.tick_params(which='both', direction='in', labelsize=tickSize)
    plt.grid()
        
    #hiding labels if required
    if hideXlabel:
        ax1.axes.get_xaxis().set_ticklabels([])
    else:
        plt.xlabel(xlabel, size=textsize)    
    if hideXticks:
        ax1.axes.get_xaxis().set_ticklabels([])
    if hideYticks:
        ax1.axes.get_yaxis().set_ticklabels([])
    if not hideYlabel:    
        plt.ylabel(ylabel, size=textsize)
    
    #Place Y axis on the right if required
    if placeYaxisOnRight:
        ax1.yaxis.tick_right()
        ax1.yaxis.set_label_position("right")
        
    #Plotting
    #define X limits if required
    if (xlim[0] is None) and (xlim[1] is None):
        rang = None
    else:
        rang = (xlim[0], xlim[1])

#     print(data, bins, integralIsOne, weights, color, align)
        
    n, bns, ptchs = plt.hist(data, bins=bins, range=rang, density=integralIsOne, weights=weights, color=color,
                             align=align, histtype=histtype, label=label, zorder=zorder, alpha=alpha,
                             cumulative=cumulative, hatch=hatch, orientation=orientation)
    
    #set hatching pattern if there is one
    
    if showLegend:
        plt.legend(loc=locLegend, prop={'size': legendTextSize}, shadow=True, fancybox=True, ncol=legendNcols)
        
    if outputName is not None:
        #If we do not want to overwrite the file
        f = None
        if not overwrite:
            #Try to open it to check if it exists
            try:
                f = open(outputName, 'r')
            except:
                pass
            if f is not None:
                print('File %s already exists but overwritting was disabled. Thus exiting without writing.' %outputName)
                return ax1, n, bns
                
        f = open(outputName, 'w')
        
        bbox_inches = None
        if tightLayout:
            bbox_inches = 'tight'
            
        plt.savefig(outputName, bbox_inches=bbox_inches)
        
    return ax1, n, bns, ptchs

                
def asManyPlots(numPlot, datax, datay, hideXlabel=False, hideYlabel=False, hideYticks=False,
                placeYaxisOnRight=False, xlabel="", ylabel='', marker='o', color='black', plotFlag=True,
                label='', zorder=0, textsize=24, showLegend=False, legendTextSize=24, linestyle='None',
                ylim=[None, None], xlim=[None, None], cmap='Greys', cmapMin=None, cmapMax=None,
                showColorbar=False, locLegend='best', tickSize=24, title='', titlesize=24, 
                colorbarOrientation='vertical', colorbarLabel=None, colorbarTicks=None, colorbarTicksLabels=None,
                colorbarLabelSize=24, colorbarTicksSize=24, colorbarTicksLabelsSize=24,
                outputName=None, overwrite=False, tightLayout=True, linewidth=3,
                fillstyle='full', unfilledFlag=False, alpha=1.0,
                noCheck=False, legendNcols=1, removeGrid=False, markerSize=16, 
                legendMarkerFaceColor=None, legendMarkerEdgeColor=None, legendLineColor=None,
                norm=None):
    """
    Function which plots on a highly configurable subplot grid either with pyplot.plot or pyplot.scatter. A list of X and Y arrays can be given to have multiple plots on the same subplot.
    This function has been developed to be used with numpy arrays or list of numpy arrays (structured or not). Working with astropy tables or any other kind of data structure might or might not work depending on its complexity and behaviour. 
    
    Input
    -----
    alpha : float, list of floats
        indicates the transparency of the data points (1 is plain, 0 is invisible)
    cmap : matplotlib colormap
        the colormap to use for the scatter plot only
    cmapMin: float
        the minmum value for the colormap
    cmapMax: float
        the maximum value for the colormap
    color : list of strings/chars/RGBs/lists of values
        color for the data. For scatter plots, the values must be in numpy array format. For plots, it can either be a string, char or RGB value.
        WARNING: it is highly recommanded to give the color as a list. For instance, if plotting only one plot of black color, you should preferentially use ['black'] rather than 'black'. For, say one plot and one scatter plot, you have to use ['black', yourNumpyArray].
    colorbarLabel : string
        the name to be put next to the colorbar
    colorbarLabelSize : int
        size of the label next to the colorbar
    colorbarOrientation : 'vertical' or 'horizontal'
        specifies if the colorbar must be place on the right or on the bottom of the graph
    colorbarTicks : list of int/float
        specifies the values taken by the ticks which will be printed next to the colorbar
    colorbarTicksLabels : list of string
        specifies the labels associated to the chosen ticks values
    colorbarTicksLabelsSize : int
        size of the labels associated to the chosen ticks
    colorbarTicksSize : int
        size of the chosen ticks
    datax: numpy array, list of numpy arrays
        the x data
    datay : numpy array, list of numpy arrays 
        the y data
    fillstyle : string, list of strings
        which fillstyle use for the markers (see matplotlib fillstyles for more information)
    hideXlabel : boolean
        whether to hide the x label or not
    hideYlabel : boolean
        whether to hide the y label or not
    hideYticks : boolean
        whether to hide the y ticks or not
    label : string
        legend label for the data
    legendLineColor : list of strings/chars/RGBs
        the line color in the legend. If None, uses the plot color (for plots) and black (for scatter plots) as default.
    legendMarkerEdgeColor : list of strings/chars/RGBs
        the color of the edges of each marker in the legend. If None, uses the plot color (for plots) and black (for scatter plots) as default.
    legendMarkerFaceColor : list of strings/chars/RGBs
        the face color (color of the main area) of each marker in the legend. If None, uses the plot color (for plots) and black (for scatter plots) as default.
    legendNcols : int
        number of columns in the legend
    legendTextSize : int
        size for the legend
    linestyle : string, list of strings for many plots
        which line style to use
    linewidth : float
        the width of the line
    locLegend : string, int
        position where to place the legend
    marker : string, char, list of both for many plots
        the marker to use for the data
    markerSize : float or list of floats for scatter plots
        the size of the marker
    noCheck : boolean
        whether to check the given parameters all have the relevant shape or not
    norm : Matplotlib Normalize instance
        the norm of the colormap (for log scale colormaps for instance)
    numPlot : int (3 digits)
        the subplot number
    outputName : str
        name of the file to save the graph into. If None, the plot is not saved into a file
    overwrite : boolean
        whether to overwrite the ouput file or not
    placeYaxisOnRight : boolean
        whether to place the y axis of the plot on the right or not
    plotFlag : boolean, list of booleans for many plots
        if True, plots with pyplot.plot function. If False, use pyplot.scatter
    removeGrid : boolean, list of booleans for many plots
        whether to remove the grid or not
    textsize : int
        size for the labels
    showColorbar : boolean
        whether to show the colorbar for a scatter plot or not
    showLegend : boolean
        whether to show the legend or not
    tickSize : int
        size of the ticks on both axes
    tightLayout : boolean
        whether to use bbox_inches='tight' if tightLayout is True or bbox_inches=None otherwise
    unfilledFlag : boolean, list of booleans
        whether to unfill the points' markers or not
    xlabel : string
        the x label
    xlim : list of floats/None
        the x-axis limits to use. If None is specified as lower/upper/both limit(s), the minimum/maximum/both values are used
    ylabel : string
        the y label
    ylim : list of floats/None
        the y-axis limits to use. If None is specified as lower/upper/both limit(s), the minimum/maximum/both values are used
    zorder : int, list of ints for many plots
        whether the data will be plot in first position or in last. The lower the value, the earlier it will be plotted
        
    Return current axis and last plot.
    """
    
    ax1 = plt.subplot(numPlot)
    ax1.yaxis.set_ticks_position('both')
    ax1.xaxis.set_ticks_position('both')
    ax1.set_title(title, size=titlesize)
    ax1.tick_params(which='both', direction='in', labelsize=tickSize)
    
    if not removeGrid:
        plt.grid(zorder=1000)
    
    #Checking shape consistency between datax and datay
    shpX = np.shape(datax)
    shpY = np.shape(datay)
    if shpX != shpY:
        exit("X data was found to have shape", shpX, "but Y data seems to have shape", shpY, ".Exiting.")
        
    #If we have an array instead of a list of arrays, transform it to the latter
    try:
        np.shape(datax[1])[0]
    except:
        datax = [datax]
        datay = [datay]
        
    #If we have only one marker/color/zorder/linestyle/label/plotFlag, transform them to a list of the relevant length
    if not noCheck:
        try:
            np.shape(linestyle)[0]
        except:
            linestyle = [linestyle]*len(datax)
    try:
        np.shape(marker)[0]
    except:
        marker = [marker]*len(datax)
    try:
        np.shape(markerSize)[0]
    except:
        markerSize = [markerSize]*len(datax)
        
    try: 
        np.shape(legendMarkerFaceColor)[0]
    except:
        legendMarkerFaceColor = [legendMarkerFaceColor]*len(datax)
        
    try: 
        np.shape(legendMarkerEdgeColor)[0]
    except:
        legendMarkerEdgeColor = [legendMarkerEdgeColor]*len(datax)
        
    try: 
        np.shape(legendLineColor)[0]
    except:
        legendLineColor = [legendLineColor]*len(datax)
        
        
    try:
        np.shape(zorder)[0]
    except:
        zorder = [zorder]*len(datax)
    try:
        np.shape(plotFlag)[0]
    except:
        plotFlag = [plotFlag]*len(datax)
    try:
        np.shape(fillstyle)[0]
    except:
        fillstyle = [fillstyle]*len(datax)
    try:
        np.shape(unfilledFlag)[0]
    except:
        unfilledFlag = [unfilledFlag]*len(datax)
    try:
        np.shape(alpha)[0]
    except:
        alpha = [alpha]*len(datax)
    try:
        np.shape(label)[0]
    except:
        if len(datax)>1:
            if showLegend:
                print("Not enough labels were given compared to data dimension. Printing empty strings instead.")
            label = ''
        label = [label]*len(datax)
    
#     print(color, marker, zorder, linestyle, plotFlag, label)
        
    #hiding labels if required
    if hideXlabel:
        ax1.axes.get_xaxis().set_ticklabels([])
    else:
        plt.xlabel(xlabel, size=textsize)    
    if hideYticks:
        ax1.axes.get_yaxis().set_ticklabels([])
    if not hideYlabel:    
        plt.ylabel(ylabel, size=textsize)
    
    #Place Y axis on the right if required
    if placeYaxisOnRight:
        ax1.yaxis.tick_right()
        ax1.yaxis.set_label_position("right")

    #Plotting
    tmp     = []
    sct     = None
    
    #list of handels for the legend
    handles = []
    
    for dtx, dty, mrkr, mrkrSz, clr, zrdr, lnstl, lbl, pltFlg, fllstl, lph, nflldFlg in zip(datax, datay, marker, markerSize, color, zorder, linestyle, label, plotFlag, fillstyle, alpha, unfilledFlag):
        edgecolor = clr
        if nflldFlg:
            facecolor = "none"
        else:
            facecolor=clr
        
        if pltFlg:
            tmp.append(plt.plot(dtx, dty, label=lbl, marker=mrkr, color=clr, zorder=zrdr, alpha=lph,
                           linestyle=lnstl, markerfacecolor=facecolor, markeredgecolor=edgecolor,
                           markersize=mrkrSz, linewidth=linewidth))
            handles.append(copy(tmp[-1][0]))
        else:     
            #Defining default bounds for scatter plot if not given
            if cmapMin is None:
                cmapMin = np.min(color[plotFlag==False])
            if cmapMax is None:
                cmapMax = np.max(color[plotFlag==False])
            
            print("marker Size", mrkrSz)
            markerObject = MarkerStyle(marker=mrkr, fillstyle=fllstl)
            sct = plt.scatter(dtx, dty, label=lbl, marker=markerObject, zorder=zrdr, 
                              cmap=cmap, norm=norm, vmin=cmapMin, vmax=cmapMax, alpha=lph, c=clr, s=mrkrSz)
            tmp.append(sct)
            
            if nflldFlg:
                sct.set_facecolor('none')
        
    if np.any(np.logical_not(plotFlag)) and showColorbar:
        col = plt.colorbar(sct, orientation=colorbarOrientation)
        col.ax.tick_params(labelsize=colorbarTicksLabelsSize)
        
        if colorbarLabel is not None:
            col.set_label(colorbarLabel, size=colorbarLabelSize)
        if colorbarTicks is not None:
            col.set_ticks(colorbarTicks)
        if colorbarTicksLabels is not None:
            if colorbarOrientation == 'vertical':
                col.ax.set_yticklabels(colorbarTicksLabels, size=colorbarTicksLabelsSize)
            elif colorbarOrientation == 'horizontal':
                col.ax.set_xticklabels(colorbarTicksLabels, size=colorbarTicksLabelsSize)
            
    if showLegend:
        
        def isOrisNotNone(data, default):
            for num, i in enumerate(data):
                if i is None:
                    data[num] = default
            return data
        
        if pltFlg:
                
        
            for h, mkfclr, mkeclr, lc, c in zip(handles, legendMarkerFaceColor, legendMarkerEdgeColor, legendLineColor, color):
                mkfclr, mkeclr, lc = isOrisNotNone([mkfclr, mkeclr, lc], c)
                
                h.set_color(lc)
                h.set_markerfacecolor(mkfclr)
                h.set_markeredgecolor(mkeclr)
            leg = plt.legend(loc=locLegend, prop={'size': legendTextSize}, shadow=True, fancybox=True, 
                             ncol=legendNcols, handles=handles)
            
        if not pltFlg:
            leg = plt.legend(loc=locLegend, prop={'size': legendTextSize}, shadow=True, fancybox=True, 
                             ncol=legendNcols)
            
            for marker, mkfclr, mkeclr, lc in zip(leg.legendHandles, legendMarkerFaceColor, legendMarkerEdgeColor, legendLineColor):
                mkfclr = isOrisNotNone([mkfclr], 'black')
                
                marker.set_color(mkfclr)
        
    #Define Y limits if required
    if ylim[0] is not None:
        ax1.set_ylim(bottom=ylim[0])
#     else:
#         ax1.set_ylim(bottom=ax.get_ylim()[0])
    if ylim[1] is not None:
        ax1.set_ylim(top=ylim[1])
#    else:
#         ax1.set_ylim(top=ax1.get_ylim()[1])
        
    #define X limits if required
    if xlim[0] is not None:
        ax1.set_xlim(left=xlim[0])
#     else:
#         ax1.set_xlim(left=ax.get_xlim()[0])
    if xlim[1] is not None:
        ax1.set_xlim(right=xlim[1])
#     else:
#         ax1.set_xlim(right=ax.get_xlim()[1])

    if outputName is not None:
        #If we do not want to overwrite the file
        f = None
        if not overwrite:
            #Try to open it to check if it exists
            try:
                f = open(outputName, 'r')
            except:
                pass
            if f is not None:
                print('File %s already exists but overwritting was disabled. Thus exiting without writing.' %outputName)
                return ax1, tmp
                
        f = open(outputName, 'w')
        
        bbox_inches = None
        if tightLayout:
            bbox_inches = 'tight'
            
        plt.savefig(outputName, bbox_inches=bbox_inches)
    
    return ax1, tmp

data = [0, 1, 2]

asManyPlots(111, data, data,
            plotFlag=[True],
            linestyle=['-'],
           # color=[data],
            cmap='winter',
            showLegend=True,
            legendMarkerFaceColor='yellow',
            legendMarkerEdgeColor='red',
            legendLineColor='black',
            markerSize=40,
            color='green',
            label='coucou')