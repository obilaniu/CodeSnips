#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Imports
import numpy                                as np


#
# Function
#

class Function(object):
	def __init__     (self, name, *args, **kwargs):
		self.name = isinstance(name, str) and name
		if not self.name:
			raise ValueError()
		
		self.setup(*args, **kwargs)
	def setup        (self, *args, **kwargs):       pass
	def dump         (self):                        pass
	def load         (self, state):                 pass
	def getParam     (self, name,  current=True):   pass
	def __call__     (self, *args, **kwargs):       pass
