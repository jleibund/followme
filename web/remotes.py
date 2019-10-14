
import json
import threading
import logging
from time import sleep
import os.path
import cv2
import base64
from tornado import httpserver, ioloop, web, websocket, options, escape
from tornado.options import define, options

import methods

cl = []
font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,500)
fontScale              = 1
fontColor              = (255,255,255)
lineType               = 2

define("port", default=8888, help="run on the given port", type=int)


class MainHandler(web.RequestHandler):

    def get(self):
        self.render('index.html')


class SocketHandler(websocket.WebSocketHandler):

    def check_origin(self, origin):
        return True

    def open(self):
        if self not in cl:
            self.reset()
            cl.append(self)
            self.send_settings()
            self.send_status()
            logging.info("WS Open")

    def on_message(self, message):
        parsed = escape.json_decode(message)
        if parsed['action'] == "get" and parsed['target'] == "status":
            self.send_status()
        if parsed['action'] == "get" and parsed['target'] == "settings":
            self.send_settings()
        elif parsed['action'] == "set" and parsed['target'] == "auto_pilot":
            self.application.vehicle.auto_pilot_index = parsed["value"]["index"]
            self.write_message(json.dumps({'ack': 'ok'}))
        elif parsed['action'] == "set" and parsed['target'] == "record":
            self.application.vehicle.record = parsed["value"]["record"]
            self.write_message(json.dumps({'ack': 'ok'}))
        elif parsed['action'] == "set" and parsed['target'] == "throttle":
            self.application.vehicle.auto_throttle = parsed["value"]["throttle"]
            self.write_message(json.dumps({'ack': 'ok'}))

    def reset(self):
        self.application.vehicle.auto_throttle = False

    def on_close(self):
        if self in cl:
            self.reset()
            cl.remove(self)
            logging.info("WS Close")

    def send_status(self):
        v = self.application.vehicle
        img64 = None
        if v.vision_sensor:
            image = v.vision_sensor.read()
            retval, encoded = cv2.imencode('.jpg', image)
            img64 = base64.b64encode(encoded).decode('ascii')
        pilot_angle = 0
        if v.pilot_angle is not None:
            pilot_angle = str(methods.angle_to_yaw(v.pilot_angle))

        pilot_throttle = 0
        if v.pilot_throttle is not None:
            pilot_throttle = str(methods.angle_to_yaw(v.pilot_throttle))

        status = {
            "image": img64,
            "controls": {
                "angle": v.pilot_angle,
                "yaw": pilot_angle,
                "throttle": pilot_throttle
                },
            "auto_pilot": {
                "auto_pilots": v.list_auto_pilot_names(),
                "index": v.auto_pilot_index},
            "sensor_reading": v.sensor_reading,
            "fps": v.fps,
            "record": v.record,
            "throttle": v.auto_throttle,
            "is_recording": v.recorder and v.recorder.is_recording,
            "f_time": v.f_time
        }
        self.write_message(json.dumps(status))

    def send_settings(self):
        settings = {
            "test": "foo"
        }
        self.write_message(json.dumps(settings))

class WebRemote(web.Application):

    def __init__(self, vehicle):
        self.vehicle = vehicle
        base_dir = os.path.dirname(__file__)
        web_dir = os.path.join(base_dir, "./frontend")
        settings = {
            'template_path': web_dir,
            'debug': True 
        }
        web.Application.__init__(self, [
            web.url(r'/', MainHandler, name="main"),
            web.url(r'/api/v1/ws', SocketHandler, name="ws"),
            web.url(r'/static/(.*)', web.StaticFileHandler, {'path': web_dir}),
        ], **settings)

    def start(self):
        self.listen(options.port)