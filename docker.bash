#!/bin/bash
if [[ $1 == "build" ]]; then
  echo "Currently not supported, using build image from the snake group"
  # docker build . -t rlearning-snake:latest --label tetris-rl-container-snake-clone 
elif [[ $1 == "start" ]]; then
  docker run -it -d --gpus 1 --name tetris-rl-container-snake-clone -v /home/Michael.Buessemeyer/tetrisRL:/tetrisRL rlearning-snake:latest
elif [[ $1 == "exec" && $2 == "-d" ]]; then
  docker exec -it -d tetris-rl-container-snake-clone ${@:3}
elif [[ $1 == "exec" ]]; then
  docker exec -it tetris-rl-container-snake-clone ${@:2} 
elif [[ $1 == "rm" ]]; then
  docker stop tetris-rl-container-snake-clone && docker rm tetris-rl-container-snake-clone
fi
