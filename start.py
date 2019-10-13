import asyncio
from sonar import StereoSonar
from camera import PiVideoStream
from imu import IMU

loop = asyncio.get_event_loop()
sonar = StereoSonar()
camera = PiVideoStream()
imu = IMU()


async def main():
    print('start sensors')
    loop.create_task(imu.start())
    loop.create_task(camera.start())
    loop.create_task(sonar.start())
    #asyncio.create_task(sensor.start())
    # loop.create_task(sonar.start())
    print('started... waiting 10 seconds')
    await asyncio.sleep(10)
    print('stop')
    # print('reading %s'%sonar.read())
    # await asyncio.sleep(1)
    # print('reading %s'%sonar.read())

#asyncio.run(main())
try:
    loop.run_until_complete(main())
finally:
    loop.close()
