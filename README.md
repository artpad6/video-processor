# video-processor

This code takes an input video and outputs a list of frame indexes that were selected, based on one or more cheap CV techniques, to be sent for further processing. The techniques are:

DIFF
Do a pixel-wise diff between this frame and the last frame selected for processing. Calculate the fraction of pixels that have changed (diff value is non-zero) and if the fraction is above a certain threshold, select this frame
Values to adjust:
- Threshold above which we send the frame

DIFF_WITH_THRESHOLDING
Do a pixel-wise diff between this frame and the last frame selected for processing. If the intensity of the diff value is small, count the diff as 0. Then calculate the non-zero pixels over the total pixels, and if that fraction is above a certain threshold, select this frame. 
Values to adjust:
- Intensity below which we set pixels of the diff to 0
- Threshold above which we send the frame

HISTOGRAM
To do

