import cv2
import numpy as np
from enum import Enum


# Types of processors
class Processor(Enum):
    DIFF = 1
    DIFF_WITH_THRESHOLDING = 2
    HISTOGRAM = 3


# Constants to adjust between runs
VIDEO_FILENAME = 'car_video.mp4'
DIFF_THRESHOLD = 0.005
PIXEL_THRESHOLD = 15
CHISQUARE_THRESHOLD = 3


# Main method for taking input video and processing frames one by one
def process_video(video_path, processor):
    # Set up video capture, get the first frame, and save properties of the video/frames
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cv2.VideoCapture.get(cap, int(cv2.CAP_PROP_FRAME_COUNT)))
    ret, frame = cap.read()
    rows, columns = frame.shape[:2]
    total_pixels = rows*columns

    # Previous frame starts as all black so we select the first
    prev_frame = np.zeros((rows, columns, 3), dtype=np.uint8)
    output_list = []
    num_frames_used = 0
    index = 0

    # Loop through frames, keeping a list of indexes of selected frames
    while cap.isOpened():
        if not ret:
            break
        index += 1
        if should_send_frame(frame, prev_frame, total_pixels, processor):
            output_list.append(index)
            num_frames_used += 1
            prev_frame = frame.copy()
        ret,frame = cap.read()
    # Calculate percentage of frames selected and write indexes of those to a txt file
    print('percentage of frames used: ' + str((num_frames_used/frame_count)*100))
    cap.release()
    output_file = output_file_name(processor)
    with open(output_file, 'w') as f:
        for item in output_list:
            f.write("%s\n" % item)


# Returns name of output file based on input video, processor, and adjustable values
def output_file_name(processor):
    if processor is Processor.DIFF:
        return f'{VIDEO_FILENAME}_DIFF_{DIFF_THRESHOLD}.txt'
    if processor is Processor.DIFF_WITH_THRESHOLDING:
        return f'{VIDEO_FILENAME}_DIFF_{DIFF_THRESHOLD} _THRESHOLD_{PIXEL_THRESHOLD}.txt'
    else:
        return f'{VIDEO_FILENAME}_HISTOGRAM_{CHISQUARE_THRESHOLD}.txt'


# Calls the appropriate processor
def should_send_frame(frame, prev_frame, total_pixels, processor):
    if processor is Processor.DIFF:
        return frame_diff(frame, prev_frame, total_pixels)
    if processor is Processor.DIFF_WITH_THRESHOLDING:
        return frame_diff_with_threshold(frame, prev_frame, total_pixels)
    if processor is Processor.HISTOGRAM:
        return compare_histogram(frame, prev_frame)
    else:
        return True


# Calculate pixel-wise diff between frames and return whether the fraction changed is above diff threshold
def frame_diff(frame, prev_frame, total_pixels):
    frame_diff = cv2.absdiff(frame, prev_frame)
    changed_pixels = cv2.countNonZero(cv2.cvtColor(frame_diff, cv2.COLOR_BGR2GRAY))
    fraction_changed = changed_pixels / total_pixels
    return fraction_changed >= DIFF_THRESHOLD


# Calculate pixel-wise diff between frames, turn all values of the diff below PIXEL_THRESHOLD to 0
# Return whether the fraction of non-zero pixels is above diff threshold
def frame_diff_with_threshold(frame, prev_frame, total_pixels):
    frame_diff = cv2.absdiff(frame, prev_frame)
    retval, frame_diff_thresholded = cv2.threshold(cv2.cvtColor(frame_diff, cv2.COLOR_BGR2GRAY), PIXEL_THRESHOLD, 255, cv2.THRESH_TOZERO)
    changed_pixels = cv2.countNonZero(frame_diff_thresholded)
    fraction_changed = changed_pixels / total_pixels
    return fraction_changed >= DIFF_THRESHOLD


# Calculate histogram for each frame and do chi-squared comparison between them
# Return whether result is above chi-square threshold
def compare_histogram(frame, prev_frame):
    # Create histograms with 256 bins
    hist = cv2.calcHist([cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)], [0], None, [256], [0, 256])
    prev_hist = cv2.calcHist([cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)], [0], None, [256], [0, 256])
    result = cv2.compareHist(hist, prev_hist, cv2.HISTCMP_CHISQR)
    return result >= CHISQUARE_THRESHOLD


# Change processor here
if __name__ == '__main__':
    process_video(video_path=VIDEO_FILENAME, processor=Processor.DIFF)
