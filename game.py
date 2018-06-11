# -*- coding: utf-8 -*-
import os
import sys
from random import randint
from curses import KEY_RIGHT, KEY_LEFT, KEY_UP, KEY_DOWN

ELEM_BLANK = 0.0
ELEM_BORDER = -1.0
ELEM_SNAKE = -0.99
ELEM_SNAKE_HEAD = -0.98
ELEM_SNAKE_TAIL = -0.97
ELEM_FOOD = 1.0

FOOD_CNT = 30
STATE_DXY = 17

ACT_FORWARD = 0
ACT_BACK = 1
ACT_LEFT = 2
ACT_RIGHT = 3

elem2chr = { ELEM_BLANK: " ", ELEM_BORDER: "#", ELEM_SNAKE: "X", ELEM_SNAKE_HEAD: "8", ELEM_SNAKE_TAIL: "O", ELEM_FOOD: "*" }

###########

class Game:
	def __init__(self, w, h):
		self.w = w
		self.h = h
		self.arr = [[ELEM_BLANK for y in range(self.h)] for x in range(self.w)] 
		self.init()

	def init(self):
		self.snake = [[7,10], [6,10], [5,10]] # Initial snake co-ordinates
		self.food = [] 
		self.gen_food()
		self.score = 0
		self.moves = 0
		self.done = False
		self.grow = False
		self.key = KEY_RIGHT # init key
		self.prevkey = self.key
		self.direction = ACT_FORWARD
		self.set_blank()
		self.add_borders()
		self.add_snake()
		self.add_food()

	def reset(self):
		self.init()
		
	def update(self):
		self.set_blank()
		self.add_borders()
		self.add_snake()
		self.add_food()
		
	def render_dxy_arr(self, arr, beg_pos, end_pos):
		for y in range(beg_pos[1], end_pos[1]):
			for x in range(beg_pos[0],end_pos[0]):
				elem = arr[x][y]
				echr = elem2chr[elem]
			#	if elem == ELEM_SNAKE_HEAD:
			#		dir2chr = { KEY_UP: "^", KEY_DOWN: "v", KEY_RIGHT: ">", KEY_LEFT: "<" }
			#		echr = dir2chr[self.key]
				sys.stdout.write(echr)
			sys.stdout.write('\n')

	def render_dxy(self, beg_pos, end_pos):
		arr = self.arr
		self.render_dxy_arr(arr, beg_pos, end_pos)

	def render_dxy_state(self):
		state = self.get_dxy_state()
		self.render_dxy_arr(state, [0, 0], [len(state), len(state[0])])

	def render(self, info = ""):
		arr = self.arr
		os.system('clear') 
		print "Score: " + str(self.score), info
		self.render_dxy([0, 0], [self.w, self.h])

	def set_blank(self):
		arr = self.arr
		h,w = self.h, self.w
		for y in range(0,h):
			for x in range(0,w):
				arr[x][y] = ELEM_BLANK

	def get_dxy_state(self, d = STATE_DXY):
		arr = self.arr
		h,w = self.h, self.w

		head_x = self.snake[0][0]
		head_y = self.snake[0][1]

		#print "head", head_x, head_y

		loc_x = 0
		loc_y = 0

		beg_x = head_x - int((d-1)/2)
		beg_y = head_y - int((d-1)/2)
		end_x = head_x + int((d-1)/2)
		end_y = head_y + int((d-1)/2)
		if beg_x < 0:
			loc_x = -beg_x
			beg_x = 0 
		if beg_y < 0:
			loc_y = -beg_y
			beg_y = 0 
		if end_x > w-1:
			end_x = w-1 
		if end_y > h-1:
			end_y = h-1

		#print "beg", beg_x, beg_y
		#print "end", end_x, end_y
		#print "loc", loc_x, loc_y

		dxy_arr = [[ELEM_BLANK for y in range(d)] for x in range(d)] 

		l_y = loc_y
		for y in range(beg_y, end_y+1):
			l_x = loc_x
			for x in range(beg_x, end_x+1):
				dxy_arr[l_x][l_y] = arr[x][y]
				l_x += 1
			l_y += 1

		# https://stackoverflow.com/questions/8421337/rotating-a-two-dimensional-array-in-python
		# rotating clockwise (forward == up)
		if self.key == KEY_UP:
			dxy_arr_rotated = dxy_arr
		elif self.key == KEY_LEFT:
			dxy_arr_rotated = zip(*dxy_arr[::-1])
			dxy_arr_rotated = zip(*dxy_arr_rotated[::-1])
			dxy_arr_rotated = zip(*dxy_arr_rotated[::-1])
		elif self.key == KEY_DOWN:
			dxy_arr_rotated = zip(*dxy_arr[::-1])
			dxy_arr_rotated = zip(*dxy_arr_rotated[::-1])
		elif self.key == KEY_RIGHT:
			dxy_arr_rotated = zip(*dxy_arr[::-1])
		
		return dxy_arr_rotated

	def get_state(self, d = STATE_DXY):
		dxy_arr_rotated = self.get_dxy_state(d)
		return dxy_arr_rotated

	def get_flattened_state(self, d = STATE_DXY):
		dxy_arr_rotated = self.get_dxy_state(d)
		state = np.reshape(dxy_arr_rotated, [1, STATE_SIZE])
		return state

	def add_borders(self):
		arr = self.arr
		h,w = self.h, self.w
		for y in range(0,h):
			for x in range(0,w):
				if (x == 0) or (x == (w-1)) or (y == 0) or (y == (h-1)): 
					arr[x][y] = ELEM_BORDER

	def add_snake(self):
		arr = self.arr
		snake = self.snake
		head = snake[0]
		tail = snake[len(snake)-1]
		for elem in snake:
			arr[elem[0]][elem[1]] = ELEM_SNAKE
		arr[tail[0]][tail[1]] = ELEM_SNAKE_TAIL
		arr[head[0]][head[1]] = ELEM_SNAKE_HEAD

	def add_food(self):
		arr = self.arr
		for elem in self.food:
			arr[elem[0]][elem[1]] = ELEM_FOOD

	def gen_food(self):
		arr = self.arr
		snake = self.snake
		self.food = []
		while len(self.food) < FOOD_CNT:
			elem = []
			while elem == []:
				elem = [randint(1, self.w-2), randint(1, self.h-2)]
				if elem in snake: 
					elem = []
				else:
					self.food.insert(0, elem)

	def move_snake(self, key):
		snake = self.snake
		self.prevkey = self.key
		self.key = key
		self.moves += 1
		pos = [snake[0][0] + (key == KEY_RIGHT and 1) + (key == KEY_LEFT and -1), snake[0][1] + (key == KEY_UP and -1) + (key == KEY_DOWN and 1)]
		snake.insert(0, pos)
		if self.grow > 1: 
			self.grow = 0 
		else:
			last = snake.pop() 

		# If snake runs over itself
		if snake[0] in snake[1:]:
			self.done = True
			return

		# If snake crosses the boundaries, make it enter from the other side
		if (snake[0][0] == 0) or (snake[0][1] == 0) or (snake[0][0] == (self.w-1)) or (snake[0][1] == (self.h-1)): 
			self.done = True
			return

		# When snake eats the food
		for i in range(0, len(self.food)):
			if snake[0] == self.food[i]:
				self.food[i] = []
				self.score += 1
				self.grow += 1
				# Calculating next food's coordinates
				while self.food[i] == []:
					self.food[i] = [randint(1, self.w-2), randint(1, self.h-2)]
					if self.food[i] in snake: self.food[i] = []

	def step(self, key):
		old_score = self.score
		self.move_snake(key)
		self.update()
		new_score = self.score
		next_state = self.get_state()
		return next_state, (new_score-old_score)*1.

