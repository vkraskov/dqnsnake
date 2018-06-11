# -*- coding: utf-8 -*-
import game
import agent
import play
import stats

import os
import sys
import time
import curses
from curses import KEY_RIGHT, KEY_LEFT, KEY_UP, KEY_DOWN
from game import ACT_FORWARD, ACT_BACK, ACT_RIGHT, ACT_LEFT

import logging
from logging.handlers import RotatingFileHandler

BUILD_NAME = "lstm.v3"

AREA_WIDTH = 60
AREA_HEIGHT = 20

EPISODES = 100001
BATCH_SIZE = 96
MAX_STEPS = 1000
ACTION_SIZE = 4
DQN_MEMSIZE = MAX_STEPS*4	# memory no less than 4 games with steps up to max steps

key2str = { KEY_UP: "up", KEY_DOWN: "down", KEY_RIGHT: "right", KEY_LEFT: "left" }
action2str = { ACT_FORWARD: "forward", ACT_BACK: "back", ACT_RIGHT: "right", ACT_LEFT: "left" }
action2key = { 
	KEY_UP:    { ACT_FORWARD: KEY_UP,    ACT_BACK: KEY_DOWN,  ACT_LEFT: KEY_LEFT,  ACT_RIGHT: KEY_RIGHT },
	KEY_RIGHT: { ACT_FORWARD: KEY_RIGHT, ACT_BACK: KEY_LEFT,  ACT_LEFT: KEY_UP,    ACT_RIGHT: KEY_DOWN  },
	KEY_DOWN:  { ACT_FORWARD: KEY_DOWN,  ACT_BACK: KEY_UP,    ACT_LEFT: KEY_RIGHT, ACT_RIGHT: KEY_LEFT  },
	KEY_LEFT:  { ACT_FORWARD: KEY_LEFT,  ACT_BACK: KEY_RIGHT, ACT_LEFT: KEY_DOWN,  ACT_RIGHT: KEY_UP    }
	}

thisName = os.path.splitext(sys.argv[0])[0]

logPath = "/tmp/"
fileName = thisName

log_formatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s] %(message)s")
logger = logging.getLogger()
logger.setLevel(logging.INFO)

log_handler = logging.handlers.RotatingFileHandler("{0}/{1}.log".format(logPath, fileName), maxBytes=(1048576*5), backupCount=7)
log_handler.setFormatter(log_formatter)
logger.addHandler(log_handler)

stats_arr = []

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
		print "render_dxy_state"
		print "----------------"
		game.render_dxy_state()
		print "----------------"

		if game.done:
			break

		time.sleep(0.25)

#############

if __name__ == "__main__":

	game = game.Game(AREA_WIDTH, AREA_HEIGHT)
	user_play(game)

	agent = agent.Agent(ACTION_SIZE, DQN_MEMSIZE)

	play = play.Play()

	stats = stats.Stats(BUILD_NAME)

	score_sum = 0.0
	time_sum = 0.0
	score_cnt = 0.0
	steps_wo_r = 0
	quality_max = 0.0

	for e in range(EPISODES):
		game.reset()
		agent.newgame()
		state = game.get_state()
		for t in range(MAX_STEPS):
			action = agent.act(state)
			key = action2key[game.key][action]
			if int(e/100)*100 == e: 
				game.render()
				print "key:", key2str[key], "    action:", action2str[action], "   time:", t
				quality = score_sum/(score_cnt+1)
				msg_str = "episode: {}/{}, epsilon: {:.2}, q: {:0.2f}, mem: {}, mem_done: {}, time: {}"\
					.format(e, EPISODES, agent.epsilon, quality, len(agent.memory), len(agent.memory_fail), time_sum/100.0)
				print msg_str
			#	print "----------------"
			#	game.render_dxy_state()
			#	print "----------------"
				time.sleep(0.05)
			next_state, reward = game.step(key)

			#if reward == 0: 
			#	steps_wo_r += 1
			#else:
			#	steps_wo_r = 0

			#if int(e/100)*100 == e: 
			#	game.render_dxy_state()
			#	print "----------------"
			#	time.sleep(0.15)
			reward = reward if not game.done else -100.0
			score_sum += game.score
			score_cnt += 1
			#print "reward", reward
			agent.remember(state, action, reward, next_state, game.done)
			state = next_state
			if game.done or steps_wo_r > 100:
				time_sum += t
				#if t > 100:
				#	game.render()
				#	time.sleep(2)
				#print("episode: {}/{}, time: {}, score: {}, e: {:.2}".format(e, EPISODES, t, game.score, agent.epsilon))
				if int(e/100)*100 == e: 
					quality = score_sum/score_cnt
					stats_arr.append((e, quality))
					msg_str = "episode: {}/{}, epsilon: {:.2}, q: {:0.2f}, mem: {}, mem_done: {}, time: {}"\
						.format(e, EPISODES, agent.epsilon, quality, len(agent.memory), len(agent.memory_fail), time_sum/100.0)
					print(msg_str)
					logger.info(msg_str)
					print("quality: {:0.2f}".format(quality))
					if quality_max < quality:
						quality_max = quality
					#	if quality_max > 15.0: 
					#		logger.info("Saving weights @ episode: {}/{}, quality: {:0.2f}".format(e, EPISODES, quality))
					#		agent.save("/tmp/" + thisName + ".h5")
					time_sum = 0.0
					score_sum = 0.0
					score_cnt = 0.0
					time.sleep(2)
				break

		stats.add(e, game.moves, game.score, game.score/100.0, agent.epsilon, len(agent.memory), len(agent.memory_fail), len(agent.memory_good))
		if int(e/100)*100 == e: 
			stats.flush()

		steps_wo_r = 0
		if len(agent.memory) > BATCH_SIZE:
			#print "agent.replay.."
			agent.replay(BATCH_SIZE)

	for elem in stats_arr:
		e = elem[0]
		v = elem[1]
		i = int(v)
		sys.stdout.write("# %-10i | %s (%0.2f)\n" % (e, '='*i, v))


