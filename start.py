import asyncio
from sonar import StereoSonar
from camera import PiVideoStream
from imu import IMU

loop = asyncio.get_event_loop()
sonar = StereoSonar()
camera = PiVideoStream()
imu = IMU()

async def main():
    print('hello')
    await asyncio.sleep(1)
    print('world')
    #asyncio.create_task(sensor.start())
    loop.create_task(sonar.start())
    await asyncio.sleep(1)
    print('reading %s'%sonar.read())
    await asyncio.sleep(1)
    print('reading %s'%sonar.read())

#asyncio.run(main())
try:
    loop.run_until_complete(imu.start())
finally:
    loop.close()
