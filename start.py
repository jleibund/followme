import asyncio
from sonar import StereoSonar

sensor = StereoSonar()

async def main():
    print('hello')
    await asyncio.sleep(1)
    print('world')
    await sensor.start()
    await sensor.update()
    print('reading %s'%sensor.read())
    await sensor.update()
    print('reading %s'%sensor.read())
    await sensor.update()
    print('reading %s'%sensor.read())

asyncio.run(main())
