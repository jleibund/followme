import asyncio
from sonar import StereoSonar
from camera import PiVideoStream
from imu import IMU
from threading import Thread

loop = asyncio.get_event_loop()
sonar = StereoSonar()
camera = PiVideoStream()
imu = IMU()

def start_loop(loop,task):
    asyncio.set_event_loop(loop)
    loop.run_until_complete(task.start())

def start_thread(task):
    loop = asyncio.new_event_loop()
    thread = Thread(target=start_loop,args=(loop,task))
    thread.start()
    return thread

async def main():
    print('start sensors')
    start_thread(imu)
    start_thread(sonar)
    loop.create_task(camera.start())
    #loop.create_task(sonar.start())
    #asyncio.create_task(sensor.start())
    # loop.create_task(sonar.start())
    print('started... waiting 10 seconds')
    await asyncio.sleep(10)
    imu.stop()
    sonar.stop()
    camera.stop()
    await asyncio.sleep(1)
    print('stop')
    # print('reading %s'%sonar.read())
    # await asyncio.sleep(1)
    # print('reading %s'%sonar.read())

#asyncio.run(main())
try:
    loop.run_until_complete(main())
finally:
    loop.close()
