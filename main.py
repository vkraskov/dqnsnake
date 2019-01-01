# -*- coding: utf-8 -*-
#import os
#import sys
#import inspect
import time
import curses
from curses import KEY_RIGHT, KEY_LEFT, KEY_UP, KEY_DOWN

import game
from game import ACT_FORWARD, ACT_BACK, ACT_RIGHT, ACT_LEFT

#BUILD_NAME = os.path.basename(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))

AREA_WIDTH = 60
AREA_HEIGHT = 20

#EPISODES = 100001
#BATCH_SIZE = 96
#MAX_STEPS = 1000
#ACTION_SIZE = 4

#key2str = { KEY_UP: "up", KEY_DOWN: "down", KEY_RIGHT: "right", KEY_LEFT: "left" }
#action2str = { ACT_FORWARD: "forward", ACT_BACK: "back", ACT_RIGHT: "right", ACT_LEFT: "left" }
#action2key = { 
#	KEY_UP:    { ACT_FORWARD: KEY_UP,    ACT_BACK: KEY_DOWN,  ACT_LEFT: KEY_LEFT,  ACT_RIGHT: KEY_RIGHT },
#	KEY_RIGHT: { ACT_FORWARD: KEY_RIGHT, ACT_BACK: KEY_LEFT,  ACT_LEFT: KEY_UP,    ACT_RIGHT: KEY_DOWN  },
#	KEY_DOWN:  { ACT_FORWARD: KEY_DOWN,  ACT_BACK: KEY_UP,    ACT_LEFT: KEY_RIGHT, ACT_RIGHT: KEY_LEFT  },
#	KEY_LEFT:  { ACT_FORWARD: KEY_LEFT,  ACT_BACK: KEY_RIGHT, ACT_LEFT: KEY_DOWN,  ACT_RIGHT: KEY_UP    }
#	}

###################

def read_stdin():
    def cb(screen):
        curses.use_default_colors()
        result = []
        screen.nodelay(1)
        while True:
            try:
                result.append(screen.getkey())
            except curses.error:
                return result

    # contains a list of keys pressed since last call
    return curses.wrapper(cb)

###################

def user_play(game):
	key = KEY_RIGHT # init key
	prevKey = key 
	while True:
		key_arr = read_stdin()
		if len(key_arr) == 0:
			key = prevKey
		elif len(key_arr) == 3:
			if key_arr[2] == 'A':
				key = KEY_UP
			if key_arr[2] == 'B':
				key = KEY_DOWN
			if key_arr[2] == 'C':
				key = KEY_RIGHT
			if key_arr[2] == 'D':
				key = KEY_LEFT
		prevKey = key
		game.move_snake(key)
		game.update()
		game.render()

		if game.done:
			break

		delay = 0.25 - game.moves*(0.25-0.05)/1000
		time.sleep(delay)

#############

if __name__ == "__main__":

	game = game.Game(AREA_WIDTH, AREA_HEIGHT)
	user_play(game)

