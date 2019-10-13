#!/bin/bash -

SCRIPT_DIR="$( cd "$( dirname "$0" )" && pwd )"
RECORD=false

while getopts ":r" opt; do
  case $opt in
    r)
      RECORD=true
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
  esac
done

if [ $RECORD = true ]; then
	sudo -E LD_LIBRARY_PATH="$LD_LIBRARY_PATH" /home/pi/.virtualenvs/openvino/bin/python $SCRIPT_DIR/drive.py --record
else
	sudo -E LD_LIBRARY_PATH="$LD_LIBRARY_PATH" /home/pi/.virtualenvs/openvino/bin/python $SCRIPT_DIR/drive.py
fi

