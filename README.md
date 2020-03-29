# Infiltration-Profiles
Refined method to binarize infiltration profiles in soils gained by dye-tracer experiments. The processing routine
automatically classifies Brilliant Blue-stained images into stained and unstained pixels using Otsu's (1979) threshold for 
image segmentation. Furthermore the routine includes some pre- and postprocessing steps like spatial correction 
(image rectification), extent trimming, and a morphological rendering and labelling of the deduced dye-patterns.

### Working example
The repository includes an Ipython notebook example and test image examplarily illustrating the procedure of image
processing. The processing scheme requires and Ipython notebook to make full use of the implemented tools.

### Required packages
The processing script requires following packages:
* cv2
* numpy
* pandas
* pickle
* tkinter
* scikit-image

### License
All codes are developed and tested with Python 3.7 and given under the GNU General Public License (GPLv3). 
Codes in this repository are provided "as is" without any warranty nor liability in any case. However, you are invited to use, 
test and modify the codes provided that the author is explicitly named.

##### Reference
Otsu, N. (1979). A threshold selection method from gray-level his-tograms. IEEE Transactions on Systems, Man, and Cybernetics,
9(1), 62–66. https://doi.org/10.1109/tsmc.1979.4310076
