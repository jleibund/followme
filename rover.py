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
from methods import min_abs, start_loop, start_thread, start_process
from multiprocessing import Queue
from threading import Thread

class Rover(object):
    '''
    Rover class
    '''

    def __init__(self):
        # initialize manual pilots
        manual_pilots = []
        manual_pilots.append(RC())
        logging.info("Loaded RC module")
        self.manual_pilots = manual_pilots
        auto_pilots = []

        # initialize autonomous pilot
        self.mobilenet = MobileNet(self)
        auto_pilots.append(self.mobilenet)
        self.auto_pilots = auto_pilots

        # add web server
        self.remote = WebRemote(self)

        # add file recorder
        self.recorder = FileRecorder(self)

        # initialize camera
        self.vision_queue = Queue()
        self.vision_sensor = PiVideoStream(self.vision_queue)

        # initialize sonar
        self.sonar_sensor = StereoSonar()

        # initialize IMU
        self.imu_sensor = IMU()

        # initialize LED
        self.indicator = NAVIO2LED()

        # initialize Ackermann
        self.mixer = AckermannSteeringMixer.from_channels()

        # init remaining
        self.auto_pilot_index = -1
        self.f_time = 0.
        self.pilot_yaw = 0.
        self.pilot_throttle = 0.
        self.record = False
        self.auto_throttle = False
        self.frame_buffer = None
        self.frame_time = None
        self.cropped_buffer = None
        self.cropped_time = None
        self.pilot_angle = None
        # holds mobilnet box when detected
        self.target = None
        self.sensor_reading = {}
        self.max_sonar_ft = volts_to_distance(config.sonar.max_mV)
        self.threshold_sonar_ft = config.sonar.threshold_sonar_ft
        self.fps = 0

    def set_indicator(self,state):
        if self.indicator:
            self.indicator.set_state(state)

    async def run(self):
        # start services
        self.set_indicator('warmup')
        start_thread([self.imu_sensor,self.sonar_sensor])
        start_process(self.vision_sensor)
        start_thread(self.mobilenet)
        self.remote.start()
        # wait and read sensors
        await asyncio.sleep(0.1)
        self.sensor_reading = self.read_sensors()
        self.set_indicator('ready')
        await asyncio.sleep(0.1)
        counter = 0

        # enter run loop
        while True:
            start_time = time.time()
            await self.step()
            stop_time = time.time()
            self.f_time = stop_time - start_time
            self.fps = 1/self.f_time
            await asyncio.sleep(max(0.002, 0.05 - self.f_time))
            counter = counter + 1


    async def step(self):
        # gather angle, throttle
        final_angle = 0.
        final_throttle = None

        # read sensors in format used by web remote and extract sonar values
        self.sensor_reading = self.read_sensors()
        sonar_left = self.sensor_reading['sonar_left']
        sonar_right = self.sensor_reading['sonar_right']
        sonar_left_ft = self.sensor_reading['sonar_left_ft']
        sonar_right_ft = self.sensor_reading['sonar_right_ft']

        # run the manual pilots, if any
        for pilot in self.manual_pilots:
            try:
                pilot_angle, pilot_throttle, frame_time = await pilot.decide()
                final_angle += pilot_angle
                final_throttle = min_abs(final_throttle, pilot_throttle)
            except Exception as e:
                pass

        # complete frame decision, if no other inputs use raw vision sensor frame
        pilot_time = time.time()
        if self.vision_sensor:
            frame_time = 0
            while pilot_time < frame_time:
                frame_buffer, frame_time = self.vision_queue.get()

        # run auto pilots
        for pilot_index in range(0,len(self.auto_pilots)):
            pilot = self.auto_pilots[pilot_index];
            if self.auto_pilot_index > -1 and pilot_index == self.auto_pilot_index:
                pilot.selected = True
                try:
                    # get computed angle, throttle and time
                    pilot_angle, pilot_throttle, frame_time = await pilot.decide()
                    self.target = pilot.target

                    # if auto pilot delay is greater than cutoff, then ignore during this step
                    if frame_time > (pilot_time-float(config.car.auto_pilot_cutoff)):
                        # add in auto pilot angle
                        final_angle += pilot_angle

                        # if auto throttle is on, use the computed throttle, else combine with other throttle inputs
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
                self.target = None


        # get the safety distance (ft) and closest sonar reading
        safety = config.car.safety
        min_sonar = min_abs(sonar_left_ft,sonar_right_ft)

        # if within the safety distance apply full reverse stop (brake)
        if min_sonar < safety:
            final_throttle = 1.0
            logging.info("Obstacle! safety STOP")
        else:
            # otherwise check for sonar factoring of throttle value
            try:
                # get max throttling distance
                throttle_sonar_ft = self.max_sonar_ft

                # if within this distance
                if min_sonar < self.threshold_sonar_ft:

                    # the sonar factor is proportion of min sonar minus the safety distance (handled above)
                    sonar_factor = (min_sonar-safety)/(self.threshold_sonar_ft-safety)

                    # if we are applying sonar throttling, multiply sonar factor by final throttle
                    if config.sonar.throttle:
                        final_throttle = sonar_factor * final_throttle

                    # if applying sonar steering (obstacle avoidance)
                    if config.sonar.steering:
                        # override the yaw by the differential between left and right sonar
                        # TODO - currently disabled, too much sonar noise to trust this computed yaw
                        yaw = (config.car.max_steering_angle) * ((sonar_right-sonar_left) / (throttle_sonar_ft)) * (1-sonar_factor)
                    # log if enabled
                    if config.mobilenet.debug:
                         logging.info("Sonar Throttling: %.3f yaw: %.3f throttle: %.3f"%(sonar_factor,yaw,final_throttle))
            except Exception as e:
                logging.error(e)

        # if no throttle yet set to 0
        if final_throttle is None:
            final_throttle = 0.0

        # save current throttle / yaw
        self.pilot_angle = final_angle
        self.pilot_throttle = final_throttle

        # send values to the mixer
        if self.mixer:
            self.mixer.update(self.pilot_throttle, self.pilot_angle)

        # if recording, exec recorder
        if self.record and self.recorder is not None:
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

    def read_sensors(self):
        ''' Builds sensor dict for use in web remote '''
        all_sensors = {}
        all_sensors['sonar_left'] = 0.0
        all_sensors['sonar_right'] = 0.0
        all_sensors['sonar_left_ft'] = 0.0
        all_sensors['sonar_right_ft'] = 0.0
        all_sensors['m9ax'] = 0.0
        all_sensors['m9ay'] = 0.0
        all_sensors['m9az'] = 0.0
        all_sensors['m9gx'] = 0.0
        all_sensors['m9gy'] = 0.0
        all_sensors['m9gz'] = 0.0
        all_sensors['m9mx'] = 0.0
        all_sensors['m9my'] = 0.0
        all_sensors['m9mz'] = 0.0

        if self.sonar_sensor:
            (sonar_left,sonar_right) = self.sonar_sensor.read()
            all_sensors['sonar_left'] = sonar_left
            all_sensors['sonar_right'] = sonar_right
            sonar_left_ft = volts_to_distance(sonar_left)
            sonar_right_ft = volts_to_distance(sonar_right)
            all_sensors['sonar_left_ft'] = sonar_left_ft
            all_sensors['sonar_right_ft'] = sonar_right_ft

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
        return all_sensors
