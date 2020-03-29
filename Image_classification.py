# coding: utf-8

# Image classification routine to extract Brilliant Blue dye-patterns from soil profile images
# This routine is given under GNU GPL3 without any liability. (cc) a.reck@tu-berlin.de 2019

import cv2 as cv
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle

from IPython.display import display
from ipywidgets import *
from matplotlib.colors import ListedColormap
from tkinter import *
from skimage.morphology import opening
from skimage.measure import label
from skimage.color import rgb2hsv
from skimage.filters import threshold_otsu
from tkinter.filedialog import askopenfilename


class BuildB(object): # Class Object
    
    def __init__(self, xlims = [0,1], ylims = [0,1]): 
        '''
        Instantiation operation, creates empty object
        '''
        self.ctrl = []                # list of control points
        self.xlims = xlims            # limits for x coordinate of control points (not outside the imge)
        self.ylims = ylims            # limits for y coordinate of control points (not outside the imge)
        self.rectified_image = []     # rectified image
        self.input_image = []         # input image
        self.directory = []           # directory of stored image
        self.file_path = []           # path of stored image
        self.file_name = []           # extracted file name 
        self.patches_stained_raw = [] # binary values of stained patches (raw)
        self.patches_stained_mod = [] # binary values of stained patches (after morphological opening and edge cut-off)
        self.hsv = []                 # rectified image in HSV-color space
        self.otsu = []                # Otsu's threshold for seperation of dye patterns
        self.otsu2 = []               # Otsu's threshold for seperation of dye patterns (if trimod=True)
        self.thresh_adj1 = []         # adjusted threshold 1
        self.thresh_adj2 = []         # adjusted threshold 2
        self.resolution = []          # resolution of binarised image
        self.structure = []           # structuring element for morphological opening
        self.trimod = []              # trimod (if trimodal hue distribution)
        self.clip_border = []         # boolean for border cut-off
        self.edge_buffer = []         # save defined width for edge cut-off
        
        
    def open_file(self, tk = True, file = None):
        '''
        Opens folder to select and import an image
        
        Parms
        ------
        tk: boolean
            if True (default) use tkinter to open file, if False file must pe specified using the file
            argument
        file: string
            file to open (only required if tkinter is not used for image import)
        '''
        if tk == True:
            root = Tk()
            root.attributes("-topmost", True) # move opening dialogue to front
            self.file_path = askopenfilename(title='Select file', 
                                         filetypes=(("jpeg files", "*.jpg"), ("all files", "*.*")))
            root.destroy() # destroy root, otherwise kernel dies
        else:
            self.file_path = file
        self.directory = os.path.split(self.file_path)[0] # directory of imported image
        temp = os.path.splitext(self.file_path)[0] # extract filename from path
        self.file_name = os.path.basename(temp)    # store file name
        
        self.input_image = plt.imread(self.file_path) # load image
        
        print ('Imported ' + self.file_name + ' from ' + self.directory)
    
    def set_points(self, event): 
        '''
        Set control points for rectification; 
        order to set points: 
        1. upper left (UL), 
        2. upper right (UR), 
        3. bottom left (BL), 
        4. bottom right (BR)
        '''
        
        if event.button==3 and event.inaxes and len(self.ctrl)<4: # set points and add to plot
            x,y = event.xdata,event.ydata
            self.ctrl.append((x,y))  
            plt.plot(x, y,  'bo', markersize=10) 
       
       
        if len(self.ctrl)==4:  # add rectangle to plot after 4th point is set
            myarray = np.asarray(self.ctrl)
            myarray1 = np.append(myarray,[myarray[0,:]],axis=0)
            idx = [0,1,3,2,0]
            myarray1 = myarray1[idx]
            plt.plot(myarray1[:,0], myarray1[:,1],color = "white", linestyle = "-", linewidth = 3)
          
       
    def plot_points(self, figure_size = [10,10]):
        '''
        Plot image and connect plot with mouse events
        
        Parms
        -----
        figure_size: ndarry
            figure size of matplotlib figure [width, height], default [10, 10]
        '''
        self.ctrl=[]
        fig = plt.figure(figsize = figure_size)
        ax = fig.add_subplot(111)
        ax.imshow(self.input_image,interpolation = "nearest") 
        ax.axis("off")
        ax.set_title('Input image: ' + self.file_name)
        words = ['1: UL', '2: UR', '3: BL', '4: BR']
        items = [widgets.Button(description = w, button_style = "danger") for w in words] # red butttons to display order of point selction
        display(HBox(items))
        ax.set_autoscale_on(False)
        fig.canvas.mpl_connect("button_press_event", self.set_points) # add points by right-clicking the mouse
        plt.show()
    
    
    
    def rectify (self, resolution, figure_size = [12,4]):
        '''
        Rectify image using selected control points
        
        Parms
        -----
        resolution: int
            resulution of rectified image [width, height] in px
        figure_size: ndarry
            figure size of matplotlib figure [width, height], default [12, 4]
        '''
        im_in = self.input_image
        points_in = np.float32([self.ctrl[0], self.ctrl[1], self.ctrl[2], self.ctrl[3]])
        points_out = np.float32([[0,0], [resolution[0],0], [0, resolution[1]], [resolution[0], resolution[1]]])
        transform_matrix = cv.getPerspectiveTransform(points_in, points_out)
        dewarp_image = cv.warpPerspective(im_in, transform_matrix, (resolution[0], resolution[1]))
        
        fig, ax = plt.subplots(1,2, figsize = figure_size)
        
        ax[0].imshow(im_in, interpolation = "nearest")
        ax[0].axis("off")
        ax[0].set_title(self.file_name + ' raw')
        
        ax[1].imshow(dewarp_image, interpolation = "nearest")
        ax[1].axis("off")
        ax[1].set_title(self.file_name + ' rectified')
        plt.show()
        
        self.rectified_image = dewarp_image   # save rectified image
        self.resolution = resolution          # save resolution
        
        
    def label_patches(self, adj_threshold1 = 0, trimod = False, adj_threshold2 = 0, structure = None, 
                      clip_borders = False, buffer = None, figure_size = [15,10]):
        '''
        Extract dye-stained patterns from rectified image by transforming image to HSV-color space and performing an
        Otsu-threshold based image segmentation on images hue channel
        
        Parms
        -----
        adj_threshold1: int
            shift threshold if tracer recovery is unsatisfying (positive values shift threshold to higher
            values, vice versa with negative values)
        trimod: boolean
            if Hue values show a trimodal distribution a second image classification can be performed
        adj_threshold2: int
            shift second threshold (in the case of a trimodal Hue distribution) if tracer recovery is unsatisfying 
            (positive values shift threshold to higher values, vice versa with negative values).  
        structure: ndarry
            structuring element used by morphological opening, dye patterns smaller this element will be removed from
            the binarised image. See function "opening" of the "scikit-image" collection for details.
        clip_borders: boolean
            whether parts of from teh edges should be removed. If True, buffer must be specified.
        bufer: int 
            width of removed border in px  (left, top, right, bottom)
        figure: ndarry
            figure size of matplotlib figure [width, height], default [15, 10]
        '''
        im_hsv = rgb2hsv(self.rectified_image)
        hue = im_hsv[:,:,0]                               # extract hue channel from image
        thresh1 = threshold_otsu(hue)                     # calculating Otsu's threshold            
        threshold1 = thresh1 + adj_threshold1             # shift threshold if desired
        markers = np.zeros(hue.shape, dtype = np.uint)    # set markers 
        if trimod == False:
            markers1 = np.zeros(hue.shape, dtype = np.uint)
            markers1[hue < threshold1] = 0
            markers1[hue > threshold1] = 1
            patches_raw = markers1
        else:
            markers2 = np.ones(hue.shape, dtype = np.uint)
            mask = hue > threshold1
            thresh2 = threshold_otsu(hue[mask])
            threshold2 = thresh2 + adj_threshold2
            markers2[hue < threshold1] = 0
            markers2[hue > threshold2] = 0
            patches_raw = markers2
            self.otsu2 = thresh2                        # save Otsu's threshold 2
            self.thresh_adj2 = threshold2               # save adjusted threshold
            
        patches_open = opening(patches_raw, structure)  # morphological opening to remove patches smaller the structuring element
        patches_labeled_open = label(patches_open)      # label identified objects (modified)
        patches_labeled_raw = label(patches_raw)        # label identified objects (raw)
       
        if clip_borders == True: # remove undesired border
            self.patches_stained_mod = patches_labeled_open[buffer[1] : self.resolution[1] - buffer[3],
                                                                 buffer[0] : self.resolution[0] - buffer[2]]
            frame = np.full([self.resolution[1], self.resolution[0]], np.nan)
            frame[: buffer[1], : ] = 1; frame[self.resolution[1] - buffer[3] :, :] = 1
            frame[:, : buffer[0] ] = 1; frame[:, self.resolution[0] - buffer[2] : ] = 1
        else:
            self.patches_stained_mod = patches_labeled_open
       
        self.patches_stained_raw = patches_labeled_raw        # save identified patches (raw)
        self.hsv = im_hsv                                     # save HSV image   
        self.otsu = thresh1                                   # save Otsu's threshold         
        self.thresh_adj1 = threshold1                         # save adjusted threshold
        self.structure = structure                            # save whether structuring element was defined (only available for strucutre != None)
        self.trimod = trimod                                  # save boolean for trimod
        self.clip_borders = clip_borders                      # save boolean for clear border
        self.edge_buffer = buffer                             # save buffer value for clear border
        
        blue = (31/255,119/255,180/255)
        colormap = ListedColormap(["white", blue])
        fig = plt.figure(figsize = figure_size)
        gs = gridspec.GridSpec(2,3)
        ax1 = plt.subplot(gs[0,0])
        ax2 = plt.subplot(gs[0,1], sharex = ax1, sharey = ax1)
        ax3 = plt.subplot(gs[0,2], sharex = ax1, sharey = ax1)
        ax4 = plt.subplot(gs[1,:])
        
        ax1.imshow(self.rectified_image)
        ax1.set_title(self.file_name + ' rectified')
        ax1.axis("off")
        
        ax2.imshow(self.patches_stained_raw, vmin = 0, vmax = 1, cmap = colormap)
        ax2.imshow(self.rectified_image, alpha = 0.3)
        ax2.set_title(self.file_name + ' identified patches (raw)')
        ax2.axis("off")
        
        ax3.imshow(patches_labeled_open, vmin = 0, vmax = 1, cmap = colormap)
        ax3.imshow(self.rectified_image, alpha = 0.3)
        if clip_borders == True:
            cmap_frame = ListedColormap(["#808080"])
            ax3.imshow(frame, cmap = cmap_frame, alpha = 0.9, vmin = 0, vmax = 1)
            ax3.set_title(self.file_name + ' identified patches\n(modified and border removed)')
        else:
            ax3.set_title(self.file_name + ' identified patches (modified)')
        ax3.axis("off")
        
        ax4.hist(np.ravel(self.hsv[:,:,0]), bins = 100, label = 'Hue values')
        ax4.axvline(self.thresh_adj1, color = "black", linestyle = "dashed", linewidth = 2, 
                    label = 'Threshold Otsu')
        if trimod == True:
            ax4.axvline(threshold2, color='black', linestyle='dotted', linewidth=2, 
                        label = 'Treshold Otsu2')
        ax4.set_title('Histogram of hue values')
        ax4.set_ylabel('Frequency')
        ax4.set_xlabel('Hue values')
        ax4.legend()
        
        plt.show()

        
    def save_rectified_image(self):
        '''
        Create subfolder rectify within the directory of the processed image and 
        save rectified image to this folder
        '''
        subfolder_rectify = self.directory + '/rectify/'
        os.makedirs(os.path.dirname(subfolder_rectify), exist_ok = True)  
        plt.imsave(subfolder_rectify+self.file_name + 'rect.png', self.rectified_image)
        print ('Image ' + self.file_name + 'rect.png saved to ' + subfolder_rectify)
    
    
    def save_labeled_patches(self):
        '''
        Create subfolder labeled patches within the directory of the processed image and 
        save labeled patches to this folder
        '''
        subfolder_patches_mod = self.directory + '/labeled_patches_modified/'
        subfolder_patches_raw = self.directory + '/labeled_patches_raw/'
        os.makedirs(os.path.dirname(subfolder_patches_mod), exist_ok = True)  
        os.makedirs(os.path.dirname(subfolder_patches_raw), exist_ok = True)
        df_mod = pd.DataFrame(self.patches_stained_mod)
        df_raw = pd.DataFrame(self.patches_stained_raw)
        df_mod.to_csv(subfolder_patches_mod + self.file_name + '_patch_mod.csv', header = False, index = False)
        df_raw.to_csv(subfolder_patches_raw+self.file_name + '_patch_raw.csv', header = False, index = False)
        print ('Labeled patches of ' + self.file_name + ' saved to ' + subfolder_patches_raw + '\nand ' + subfolder_patches_mod)
        
        
    def save_object (self):
        '''
        Create subfolder pickles within the directory of the processed image and 
        save class object as pickle
        '''
        res = {}     # create dictionary to pickle specifications during processing (kind of back up)
        res['points_rectify'] = self.ctrl
        res['threshold_otsu'] = self.otsu
        res['threshold_otsu2'] = self.otsu2
        res['adjusted_threshold1'] = self.thresh_adj1
        res['adjusted_threshold2'] = self.thresh_adj2
        res['structuring_element'] = self.structure
        res['trimod'] = self.trimod
        res['clip_border'] = self.clip_border
        res['edge_buffer'] = self.edge_buffer
        
        subfolder_pickles = self.directory + '/pickles/'
        os.makedirs(os.path.dirname(subfolder_pickles), exist_ok = True)
        with open (subfolder_pickles+self.file_name + '.pickle', 'wb') as file:
            pickle.dump(res, file)
        print('File ' + self.file_name + '.pickle saved to ' + subfolder_pickles)