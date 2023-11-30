import numpy as np
import cv2
import os

__all__ = [
    'RsCamera',
    'DatasetCamera',
]

class RsCamera:
    def __init__(self):

        import pyrealsense2 as rs

        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        config = rs.config()

        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))

        found_rgb = False
        for s in device.sensors:
            if s.get_info(rs.camera_info.name) == 'RGB Camera':
                found_rgb = True
                break
        if not found_rgb:
            raise RuntimeError("Couldn't find RGB...")

        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        if device_product_line == 'L500':
            config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
        else:
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # Start streaming
        self.pipeline.start(config)

    def get_rgb(self):

        # Wait for a coherent pair of frames: depth and color
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        # depth_frame = frames.get_depth_frame()

        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        # depth_frame = frames.get_depth_frame()

        img = cv2.cvtColor(np.asanyarray(color_frame.get_data()), cv2.COLOR_BGR2RGB)
        # img = np.asanyarray(color_frame.get_data())

        return img


class DatasetCamera:
    def __init__(self, path: str, image_name='rgb.png'):
        
        self.data_samples = [os.path.join(path, i) for i in list(sorted(os.listdir(path)))
                             if os.path.isdir(os.path.join(path, i))]

        self.image_name = image_name
        self.idx = 0
        self.__current_img = None
        self.__update_current_image()


    def get_rgb(self):

        ch = cv2.waitKey(200)
        if ch == 13:  # enter
            self.idx = (self.idx + 1) % len(self.data_samples)
            self.__update_current_image()
        return self.__current_img
        
    def __update_current_image(self):
        self.__current_img = cv2.cvtColor(cv2.imread(os.path.join(self.data_samples[self.idx], self.image_name)), cv2.COLOR_BGR2RGB)