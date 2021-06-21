#!/bin/bash
if [[ $1 == "build" ]]; then
  docker build . -t tetris-rl-container:latest --label tetris-rl-container
elif [[ $1 == "start" ]]; then
  docker run -it -d --gpus 1 --name tetris-rl-container -v /home/Michael.Buessemeyer/tetrisRL:/tetrisRL tetris-rl-container:latest
elif [[ $1 == "exec" ]]; then
  docker exec -it tetris-rl-container bash
elif [[ $1 == "rm" ]]; then
  docker stop tetris-rl-container && docker rm tetris-rl-container
fi
