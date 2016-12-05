#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Imports
import numpy                                as np
import theano                               as T
import theano.tensor                        as TT
import theano.tensor.nnet                   as TTN
import theano.tensor.nnet.conv              as TTNC
import theano.tensor.nnet.bn                as TTNB
import theano.tensor.signal.pool            as TTSP
from   theano import config                 as TC
import theano.printing                      as TP


#
# Theano Function
#

class TheanoFunction(Function):
	def setup(self):
		self.optimizerStates = []
	
	def pre(self):
		for os in self.optimizerStates:
			os.pre()
	
	def post(self):
		for os in self.optimizerStates:
			os.post()
	