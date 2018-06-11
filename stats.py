# -*- coding: utf-8 -*-
from influxdb import InfluxDBClient
import datetime
import numpy as np

influx_client = InfluxDBClient("localhost", 8086, "todd", "influxdb41yf3", "test")

class Stats:
	def __init__(self):
		self.arr_json = []

	def add(self, episode, steps, score, quality, epsilon, mem, mem_fail, mem_good):
		dt = datetime.datetime.now()
		ts = dt.strftime('%s')

		self.arr_json.append(
			{
			"measurement": "dqnsnake",
			"tags": { "episode": episode },
			"time": int(ts) *1000*1000*1000,
			"fields": { 
				"steps": steps, "score": score, 
				"quality": quality, "epsilon": epsilon, 
				"mem": mem, "mem_fail": mem_fail, "mem_good": mem_good
				}
			}
		)

	def flush(self):
		influx_client.write_points(np.array(self.arr_json).tolist())
		self.arr_json = []


