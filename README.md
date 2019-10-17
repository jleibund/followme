# Autonomous "Follow-Me" Rover
![Tekno EB410 "Follow-Me" Rover](https://github.com/jleibund/followme/blob/dev/rover_side.jpg?raw=true)

## Parts List
- Tekno EB410 Rover
- Google Coral EdgeTPU
- RPi 3B+ (soon 4)
- Emlid NAVIO2
- MaxBotix MB1240 EZ4 Ultrasonic Distance Sensors (x2)
- Castle Creations Sidewinder SCT ESC
- Jrelecs 540 17.5T 2400KV Sensored Motor
- Arducam M12 Lenses (120 degree FOV)
- 4000mAh Auxillary RPi Battery Pack
- ALFA AWUS036NEH Long Range WIRELESS 802.11b/g/n Wi-Fi USBAdapter
- Miuzei Raspberry Pi 3 Model B+ Case
- JConcepts EB410 Carbon Fiber Offset Upper Deck Fan Mount (mounting to case)
- TBR Bastion Front Bumper - Tekno RC EB410
- Radiolink T8FB Transmitter (SBUS)
- 3MM Plexiglass Sheet / Cutter

## Install Instructions

- Build the vehicle.  Obtain RPi 3b+ (or 4 when EMLID image is available), Camera, Wifi AP, Google Coral USB EdgeTPU
- Use EMLID NAVIO2 Raspian Image, Complete Hardware Setup & Installation:  https://docs.emlid.com/navio2/ardupilot/hardware-setup/
- Complete Motor, Servo and Receiver Wiring (use an SBUS compatible receiver)
- Complete Sonar wiring harness and connections:  https://www.youtube.com/watch?v=Rba1ZdL0vyE&feature=youtu.be
- Setup AP on Raspberry Pi:  https://www.raspberrypi.org/documentation/configuration/wireless/access-point.md
- Install/Setup PiCamera on Raspberry Pi:  https://www.raspberrypi.org/documentation/configuration/camera.md
- Complete Google Coral Setup:  https://coral.withgoogle.com/docs/accelerator/get-started/
- Run install.sh from this repo AND/OR follow opencv install instructions:  https://www.pyimagesearch.com/2019/09/16/install-opencv-4-on-raspberry-pi-4-and-raspbian-buster/

## Running

You can start the service interactively using

``` sudo ./service.sh ```

When it is not running as a service.

## Service Commands

Use start.sh and stop.sh to start and stop as a systemd service

## Logs

You can tail syslog

``` tail -f /var/log/syslog ```
