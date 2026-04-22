.PHONY: build run

SHELL = /bin/sh

USER_ID := $(shell id -u)
GROUP_ID := $(shell id -g)

GPUS ?= 0
MACHINE ?= default

build:
	COMPOSE_DOCKER_CLI_BUILD=1 docker --context $(MACHINE) compose build mpings --build-arg USER_ID=$(USER_ID) --build-arg GROUP_ID=$(GROUP_ID)

run:
	docker --context $(MACHINE) compose run --rm -e CUDA_VISIBLE_DEVICES=$(GPUS) mpings bash
	
up:
	docker --context $(MACHINE) compose up -d mpings

exec:
	docker --context $(MACHINE) compose exec -e CUDA_VISIBLE_DEVICES=$(GPUS) mpings bash

down:
	docker --context $(MACHINE) compose down

# type exit to exit the bash shell

