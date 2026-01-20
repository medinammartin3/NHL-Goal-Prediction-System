#!/bin/bash

echo "docker-compose up --build"

docker build -t ift6758_flask_server -f Dockerfile.serving .