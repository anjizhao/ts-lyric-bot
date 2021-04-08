#!/bin/sh

. env/bin/activate
PYTHONPATH=./ python code/bot/ts_lyric_bot.py
deactivate
