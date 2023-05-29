import numpy as np
import multiprocessing as mp
import cv2
import hl2ss
import hl2ss_mp

class Streamer:
    def __init__(self):
        # Params --------------------------------------------------------------------
        self.host = '192.168.0.100'

        # Ports
        self.ports = [
            hl2ss.StreamPort.PERSONAL_VIDEO
        ]
        draw_fixation_points = True
        # PV parameters
        self.pv_mode = hl2ss.StreamMode.MODE_1
        self.pv_width = 960
        self.pv_height = 540
        self.pv_framerate = 30
        self.pv_profile = hl2ss.VideoProfile.H265_MAIN
        self.pv_bitrate = 5 * 1024 * 1024
        self.pv_format = 'bgr24'

        # Maximum number of frames in buffer
        self.buffer_elements = 1

        print(self.pv_width)
        self.client_rc = hl2ss.ipc_rc(self.host, hl2ss.IPCPort.REMOTE_CONFIGURATION)
        hl2ss.start_subsystem_pv(self.host, hl2ss.StreamPort.PERSONAL_VIDEO)
        self.calibration = hl2ss.download_calibration_pv(self.host, hl2ss.StreamPort.PERSONAL_VIDEO, self.pv_width, self.pv_height,
                                                         self.pv_framerate)
        self.client_rc.wait_for_pv_subsystem(True)

        self.producer = hl2ss_mp.producer()
        self.producer.configure_si(self.host, hl2ss.StreamPort.SPATIAL_INPUT, hl2ss.ChunkSize.SPATIAL_INPUT)
        self.producer.initialize(hl2ss.StreamPort.SPATIAL_INPUT, hl2ss.Parameters_SI.SAMPLE_RATE * 5)
        self.producer.start(hl2ss.StreamPort.SPATIAL_INPUT)
        self.manager = mp.Manager()
        self.consumer = hl2ss_mp.consumer()
        self.producer.configure_pv(True, self.host, hl2ss.StreamPort.PERSONAL_VIDEO, hl2ss.ChunkSize.PERSONAL_VIDEO, self.pv_mode,
                              self.pv_width, self.pv_height, self.pv_framerate, self.pv_profile, self.pv_bitrate, self.pv_format)
        self.producer.initialize(self.ports[0], self.buffer_elements)
        self.producer.start(self.ports[0])

        self.sink_si = self.consumer.create_sink(self.producer, hl2ss.StreamPort.SPATIAL_INPUT, self.manager, None)
        self.sink_si.get_attach_response()
        self.sink_pv = self.consumer.create_sink(self.producer, self.ports[0], self.manager, None)
        self.sink_pv.get_attach_response()

    def get_frame_and_gaze(self):
        data_pv = self.sink_pv.get_most_recent_frame()
        if not data_pv:
            return False, None, None
        data_si = self.sink_si.get_nearest(data_pv.timestamp)[1]
        gaze_point = self.extract_gaze(data_pv, data_si)
        return True, data_pv.payload.image, gaze_point

    def extract_gaze(self, data_pv, data_si):
        # self.result.write(payload.image)
        # projection = hl2ss_3dcv.projection(self.calibration.intrinsics, hl2ss_3dcv.world_to_reference(data_pv.pose))
        si = hl2ss.unpack_si(data_si.payload)
        K = np.array([[data_pv.payload.focal_length[0], 0, data_pv.payload.principal_point[0]],
                      [0, data_pv.payload.focal_length[1], data_pv.payload.principal_point[1]],
                      [0, 0, 1]])
        eye_ray = si.get_eye_ray()
        eye_ray.origin
        eye = eye_ray.direction
        pose = (data_pv.pose)
        rvec, _ = cv2.Rodrigues(pose[:3, :3])
        tvec = pose[:3, 3]
        xy, _ = cv2.projectPoints(eye, rvec, tvec, K, None)
        ixy = (int(xy[0][0][0]), int(xy[0][0][1]))
        return ixy