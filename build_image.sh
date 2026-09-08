#!/bin/bash
readonly SCRIPT_DIR=$(dirname "$(readlink -f "${BASH_SOURCE}"c)")
DOCKER_REGISTRY=pharos
IMAGE_NAME=${DOCKER_REGISTRY}/pharos
TAG=multiclass
BASE_IMG=nvidia/cuda:12.8.0-runtime-ubuntu22.04

DOCKERFILE=Dockerfile

docker build -t $IMAGE_NAME:$TAG \
    -f $DOCKERFILE \
    --network host \
    --build-arg BASE_IMG=$BASE_IMG \
    ${SCRIPT_DIR}
