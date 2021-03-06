import argparse
import configparser
import json
import logging
import imutils
from enum import Enum
from pathlib import Path

import cv2
import numpy as np


# Types of processors
class Processor(Enum):
    DIFF = 1
    DIFF_WITH_THRESHOLDING = 2
    HISTOGRAM = 3
    EDGE_DIFF = 4
    MIN_MOTION_AREA = 5


class DiffProcessor:

    def should_send_frame(self, frame, prev_frame, total_pixels):
        return True

    def process_video(self, video_path):
        # Set up video capture, get the first frame, and save properties of the video/frames
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cv2.VideoCapture.get(cap, int(cv2.CAP_PROP_FRAME_COUNT)))
        ret, frame = cap.read()
        rows, columns = frame.shape[:2]
        total_pixels = rows * columns

        # Previous frame starts as all black so we select the first
        prev_frame = np.zeros((rows, columns, 3), dtype=np.uint8)
        output_list = []
        num_frames_used = 0
        index = 0

        start_time = cv2.getTickCount()

        # Loop through frames, keeping a list of indexes of selected frames
        while cap.isOpened():
            if not ret:
                break
            index += 1

            if self.should_send_frame(frame, prev_frame, total_pixels):
                output_list.append(index)
                num_frames_used += 1
                prev_frame = frame.copy()

            ret, frame = cap.read()

        end_time = cv2.getTickCount()
        seconds_per_frame = ((end_time - start_time) / cv2.getTickFrequency())/frame_count

        # Calculate percentage of frames selected and write indexes of those to a json file
        cap.release()
        logging.info(f'{self.section}@{video_path}: {(num_frames_used/frame_count)*100:.2f}% ({num_frames_used}/{frame_count}), {seconds_per_frame:.2f} sec/frame')
        return {
            'selected_frames': output_list,
            'total_frame_count': frame_count,
            'selected_frame_count': num_frames_used,
            'seconds_per_frame': seconds_per_frame,
        }


class SimpleDiff(DiffProcessor):

    def __init__(self, section):
        self.section = section.name
        self.diff_thresh = section.getfloat('DiffThresh')

    def should_send_frame(self, frame, prev_frame, total_pixels):
        frame_diff = cv2.absdiff(frame, prev_frame)
        changed_pixels = cv2.countNonZero(cv2.cvtColor(frame_diff, cv2.COLOR_BGR2GRAY))
        fraction_changed = changed_pixels / total_pixels
        return fraction_changed >= self.diff_thresh


class DiffWithPixel(DiffProcessor):

    def __init__(self, section):
        self.section = section.name
        self.diff_thresh = section.getfloat('DiffThresh')
        self.pixel_thresh = section.getint('PixelThresh')

    def should_send_frame(self, frame, prev_frame, total_pixels):
        frame_diff = cv2.absdiff(frame, prev_frame)
        _, frame_diff_thresholded = cv2.threshold(cv2.cvtColor(frame_diff, cv2.COLOR_BGR2GRAY),
                                                  self.pixel_thresh,
                                                  255, cv2.THRESH_TOZERO)
        changed_pixels = cv2.countNonZero(frame_diff_thresholded)
        fraction_changed = changed_pixels / total_pixels
        return fraction_changed >= self.diff_thresh


class Histogram(DiffProcessor):

    def __init__(self, section):
        self.section = section.name
        self.chi_square_thresh = section.getint('ChiSquareThresh')

    def should_send_frame(self, frame, prev_frame, total_pixels):
        return self.compare_histogram(frame, prev_frame)

    def compare_histogram(self, frame, prev_frame):
        # Create histograms with 256 bins
        hist = cv2.calcHist([cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)], [0], None, [256], [0, 256])
        prev_hist = cv2.calcHist([cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)], [0], None, [256], [0, 256])
        result = cv2.compareHist(hist, prev_hist, cv2.HISTCMP_CHISQR)
        return result >= self.chi_square_thresh


class EdgeDiff(DiffProcessor):

    def __init__(self, section):
        self.section = section.name
        self.diff_thresh = section.getfloat('DiffThresh')
        self.min_val = section.getint('MinVal')
        self.max_val = section.getint('MaxVal')

    def should_send_frame(self, frame, prev_frame, total_pixels):

        # Calculate edges using tunable parameters (at what intensity is it considered an edge)?
        # Note: this is for testing the accuracy; when implementing, should save prev_frame's edges instead
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_filtered = cv2.bilateralFilter(gray, 7, 50, 50)
        edges_filtered = cv2.Canny(gray_filtered, self.min_val, self.max_val)

        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        prev_gray_filtered = cv2.bilateralFilter(prev_gray, 7, 50, 50)
        prev_edges_filtered = cv2.Canny(prev_gray_filtered, self.min_val, self.max_val)

        # Compare the edges of each frame
        frame_diff = cv2.absdiff(edges_filtered, prev_edges_filtered)
        changed_pixels = cv2.countNonZero(frame_diff)
        fraction_changed = changed_pixels / total_pixels
        return fraction_changed >= self.diff_thresh


class MinMotionArea(DiffProcessor):

    def __init__(self, section):
        self.section = section.name
        self.min_area_fraction = section.getfloat('MinArea')

    def should_send_frame(self, frame, prev_frame, total_pixels):
        # Convert frames to gray and blur them
        # Note: this is for testing the accuracy; when implementing, should save prev_frame blurred instead
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.GaussianBlur(prev_gray, (21, 21), 0)

        frame_delta = cv2.absdiff(gray, prev_gray)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]

        thresh = cv2.dilate(thresh, None, iterations=2)
        contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)

        # loop over the contours
        area_found = False
        for c in contours:
            # send frame only if a contour area was larger than min
            if (cv2.contourArea(c)/total_pixels) > self.min_area_fraction:
                area_found = True
                break
        return area_found


def init_diff_processor(section: configparser.SectionProxy):
    """Returns a DiffProcessor object specified by `section`."""
    processor_id = section.getint('Processor')
    if Processor(processor_id) == Processor.DIFF:
        return SimpleDiff(section)
    elif Processor(processor_id) == Processor.DIFF_WITH_THRESHOLDING:
        return DiffWithPixel(section)
    elif Processor(processor_id) == Processor.HISTOGRAM:
        return Histogram(section)
    elif Processor(processor_id) == Processor.EDGE_DIFF:
        return EdgeDiff(section)
    elif Processor(processor_id) == Processor.MIN_MOTION_AREA:
        return MinMotionArea(section)
    raise NotImplementedError


def _make_parser():
    _parser = argparse.ArgumentParser(description='Generates stripped video based on different diff processors.')
    _parser.add_argument('-v', '--video_path', default='data/video/auburn.mp4', help='video path')
    _parser.add_argument('-d', '--video_dir', default='', help='directory of videos')
    _parser.add_argument('-o', '--output_dir', default='', help='output json path')
    _parser.add_argument('-c', '--config', default='diff_setting.config', help='config file path')
    _parser.add_argument('--verbose', action='store_true', help='verbose mode')
    return _parser


# Change processor here
if __name__ == '__main__':
    # Parses command line arguments, for video path
    parser = _make_parser()
    args = parser.parse_args()
    # Verbose mode: sets logging config
    if args.verbose:
        logging.basicConfig(format='%(asctime)s %(levelname)s [%(funcName)s] %(message)s',
                            datefmt='%y/%m/%d %H:%M:%S',
                            level=logging.INFO)
    # Parses config file for configurations of diff processors
    config = configparser.ConfigParser()
    config.read(args.config)

    # If args.video_directory is specified, processes on all videos located in
    # that directory, ignores args.video_path'
    if args.video_dir != '':
        video_dir = Path(args.video_dir)
        videos = [str(file) for file in video_dir.iterdir() if file.suffix == '.mp4']
    else:
        videos = [args.video_path]

    # Generates a single json file containing all results regarding to all
    # processors
    selected_frames = {
        video: {
            section: init_diff_processor(config[section]).process_video(video_path=video)
            for section in config.sections()
        }
        for video in videos
    }

    # Writes output to local directory, one json file for one video
    if args.output_dir != '':
        for video_name, result in selected_frames.items():
            video_json_name = f'{str(Path(video_name).stem)}.json'
            video_json_path = Path(args.output_dir) / video_json_name
            with open(video_json_path, 'w') as j:
                j.write(json.dumps(result))
            logging.info(f'results for {video_name} saved to: {video_json_path}')