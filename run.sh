#!/bin/bash

echo "TODO: fill in the docker run command"

source .env
docker run -d -p 5000:5000 --name flask_server -e WANDB_API_KEY=$WANDB_API_KEY ift6758_flask_server