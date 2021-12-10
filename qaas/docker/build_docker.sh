#!/usr/bin/env bash

cd ..
docker build . -f docker/Dockerfile --no-cache -t autoqaas:v1.0
