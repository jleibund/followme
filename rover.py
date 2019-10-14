import sys
import time
import logging
import json
import traceback
import asyncio
from config import config
from filerecorder import FileRecorder
from sensors import StereoSonar, IMU, PiVideoStream, volts_to_distance
from pilots import RC, MobileNet
from ackermann import AckermannSteeringMixer
from indicators import NAVIO2LED
from web import WebRemote
from methods import min_abs

def start_loop(loop,task):
    if task is None:
        return
    asyncio.set_event_loop(loop)
    loop.run_until_complete(task.start())

def start_thread(task):
    if task is None:
        return
    loop = asyncio.new_event_loop()
    thread = Thread(target=start_loop,args=(loop,task))
    thread.start()
    return thread

class Rover(object):
    '''
    Rover class
    '''

    def __init__(self):
        manual_pilots = []
        manual_pilots.append(RC())
        logging.info("Loaded RC module")
        self.manual_pilots = manual_pilots

        auto_pilots = []
        self.mobilenet = MobileNet(rover,"/home/pi/byob/byob/detect_edgetpu.tflite")
        auto_pilots.append(self.mobilenet)
        self.auto_pilots = auto_pilots

        self.remove = WebRemote()
        self.recorder = FileRecorder(self)
        self.auto_pilot_index = -1
        self.f_time = 0.
        self.pilot_yaw = 0.
        self.pilot_throttle = 0.
        self.record = False
        self.auto_throttle = False
        self.vision_sensor = PiVideoStream()
        self.sonar_sensor = StereoSonar()
        self.imu_sensor = IMU()
        self.indicator = NAVIO2LED()
        self.mixer = AckermannSteeringMixer.from_channels()
        self.pilot_angle = None
        self.detection = None
        self.base64_frame = None
        self.sensor_reading = {}
        self.sensor_reading['sonar_left'] = 0.0
        self.sensor_reading['sonar_right'] = 0.0
        self.sensor_reading['sonar_left_ft'] = 0.0
        self.sensor_reading['sonar_right_ft'] = 0.0
        self.max_sonar_ft = volts_to_distance(config.sonar.max_mV)
        self.threshold_sonar_ft = config.sonar.threshold_sonar_ft
        self.fps = 0

    def set_indicator(self,state):
        if self.indicator:
            self.indicator.set_state(state)

    async def run(self):
        self.set_indicator('warmup')
        await asyncio.sleep(0.5)
        start_thread(self.imu_sensor)
        start_thread(self.sonar_sensor)
        start_thread(self.vision_sensor)
        start_thread(self.mobilenet)
        await asyncio.sleep(0.5)
        self.remote.start()
        self.set_indicator('ready')
        await asyncio.sleep(0.5)
        counter = 0
        while True:
            start_time = time.time()
            await self.step()
            stop_time = time.time()
            self.f_time = stop_time - start_time
            self.fps = 1/self.f_time
            await asyncio.sleep(max(0.002, 0.05 - self.f_time))
            counter = counter + 1

    async def step(self):
        final_angle = 0.
        final_throttle = None

        all_sensors = {}

        sonar_left = 0
        sonar_right = 0
        sonar_left_ft = 0
        sonar_right_ft = 0
        if self.sonar_sensor:
            (sonar_left,sonar_right) = self.sonar_sensor.read()
            all_sensors['sonar_left'] = sonar_left
            all_sensors['sonar_right'] = sonar_right
            sonar_left_ft = volts_to_distance(sonar_left)
            sonar_right_ft = volts_to_distance(sonar_right)
            all_sensors['sonar_left_ft'] = sonar_left_ft
            all_sensors['sonar_right_ft'] = sonar_right_ft
        else:
            all_sensors['sonar_left'] = 0.0
            all_sensors['sonar_right'] = 0.0
            all_sensors['sonar_left_ft'] = 0.0
            all_sensors['sonar_right_ft'] = 0.0

        if self.imu_sensor:
            (m9a, m9g, m9m) = self.imu_sensor.read()
            all_sensors['m9ax'] = m9a[0]
            all_sensors['m9ay'] = m9a[1]
            all_sensors['m9az'] = m9a[2]
            all_sensors['m9gx'] = m9g[0]
            all_sensors['m9gy'] = m9g[1]
            all_sensors['m9gz'] = m9g[2]
            all_sensors['m9mx'] = m9m[0]
            all_sensors['m9my'] = m9m[1]
            all_sensors['m9mz'] = m9m[2]
        else:
            all_sensors['m9ax'] = 0.0
            all_sensors['m9ay'] = 0.0
            all_sensors['m9az'] = 0.0
            all_sensors['m9gx'] = 0.0
            all_sensors['m9gy'] = 0.0
            all_sensors['m9gz'] = 0.0
            all_sensors['m9mx'] = 0.0
            all_sensors['m9my'] = 0.0
            all_sensors['m9mz'] = 0.0

        self.sensor_reading = all_sensors

        for pilot in self.manual_pilots:
            try:
                pilot_angle, pilot_throttle, frame_time = await pilot.decide()
                final_angle += pilot_angle
                final_throttle = min_abs(final_throttle, pilot_throttle)
            except Exception as e:
                pass

        pilot_time = time.time()

        min_sonar = min_abs(sonar_left_ft, sonar_right_ft)
        for pilot_index in range(0,len(self.auto_pilots)):
            pilot = self.auto_pilots[pilot_index];
            if self.auto_pilot_index > -1 and pilot_index == self.auto_pilot_index:
                pilot.selected = True
                try:
                    pilot_angle, pilot_throttle, frame_time = await pilot.decide()
                    if frame_time > (pilot_time-float(config.car.auto_pilot_cutoff)):
                        final_angle += pilot_angle
                        if self.auto_throttle:
                            final_throttle = -pilot_throttle
                        else:
                            final_throttle = min_abs(final_throttle, pilot_throttle)
                    else:
                        logging.info("Autopilot cutoff! %.3f"%(frame_time-pilot_time-config.car.auto_pilot_cutoff))
                except Exception as e:
                    logging.error("Auto pilots could not decide: %s"%e)
                    pass
            else:
                pilot.selected = False

        safety = config.car.safety
        min_sonar = min_abs(sonar_left_ft,sonar_right_ft)
        if min_sonar < safety:
            final_throttle = 1.0
            logging.info("Obstacle! safety STOP")
        else:
            try:
                throttle_sonar_ft = self.max_sonar_ft
                if min_sonar < self.threshold_sonar_ft:
                    sonar_factor = (min_sonar-safety)/(self.threshold_sonar_ft-safety)
                    if config.sonar.throttle:
                        final_throttle = sonar_factor * final_throttle
                    if config.sonar.steering:
                        yaw = (config.car.max_steering_angle) * ((sonar_right-sonar_left) / (throttle_sonar_ft)) * (1-sonar_factor)
                    if config.mobilenet.debug:
                         logging.info("Sonar Throttling: %.3f yaw: %.3f throttle: %.3f"%(sonar_factor,yaw,final_throttle))
            except Exception as e:
                logging.error(e)

        if final_throttle is None:
            final_throttle = 0.0

        self.pilot_angle = final_angle
        self.pilot_throttle = final_throttle

        if self.mixer:
            self.mixer.update(self.pilot_throttle, self.pilot_angle)

        if self.record:
            self.recorder.record_frame()

        if self.recorder and self.recorder.is_recording:
            self.set_indicator('recording')
        elif self.record:
            self.set_indicator('standby')
        else:
            self.set_indicator('ready')

    def pilot(self):
        if self.auto_pilot_index >= 0:
            return self.auto_pilots[self.auto_pilot_index]
        return None

    def list_auto_pilot_names(self):
        return [p.pname() for p in self.auto_pilots]
