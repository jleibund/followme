import asyncio
from sonar import StereoSonar

loop = asyncio.get_event_loop()
sensor = StereoSonar()

async def main():
    print('hello')
    await asyncio.sleep(1)
    print('world')
    #asyncio.create_task(sensor.start())
    loop.create_task(sensor.start())
    await asyncio.sleep(1)
    print('reading %s'%sensor.read())
    await asyncio.sleep(1)
    print('reading %s'%sensor.read())

#asyncio.run(main())
try:
    loop.run_until_complete(sensor.start())
finally:
    loop.close()
