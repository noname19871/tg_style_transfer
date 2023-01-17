#!/bin/bash

exec python -u bot.py & python -u redis_worker.py