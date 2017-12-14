#!/usr/bin/env python
# -*- coding: utf-8 -*-

#
# Imports
#

import numpy                                as np
import torch                                as T
import torch.autograd                       as TA
import torch.cuda                           as TC
import torch.nn                             as TN
import torch.optim                          as TO

from   torch                            import (DoubleTensor)
from   torch.autograd                   import (Variable)
from   torch.nn                         import (Module)



class GradProbe(Module):
	def __init__(self, fwMonitor=True, bwMonitor=True):
		super(GradProbe, self).__init__()
		
		self.fwMonitor = fwMonitor
		self.bwMonitor = bwMonitor
		self.register_buffer("fwSqSum", DoubleTensor(1).zero_())
		self.register_buffer("bwSqSum", DoubleTensor(1).zero_())
		self.register_backward_hook(self.bwHook)
	
	def forward(self, arg):
		if self.fwMonitor:
			self.fwSqSum.add_(Variable(arg.data, volatile=True).norm().double().data**2)
		return arg
	
	def bwHook(self, grI, grO):
		if self.bwMonitor:
			self.bwSqSum.add_(Variable(grI.data, volatile=True).norm().double().data**2)
		return grI
	
	def zero_grad(self):
		self.zero_fw().zero_bw()
	
	def zero_fw(self):
		self.fwSqSum.zero_()
		return self
	
	def zero_bw(self):
		self.bwSqSum.zero_()
		return self
	
	#
	# Defang most type converters in order to keep our buffers as ultra-high-resolution DoubleTensors.
	#
	# Do however allow the module's buffers to be migrated CPU <-> GPU.
	#
	
	def type  (*args): pass
	def half  (*args): pass
	def float (*args): pass
	def double(*args): pass

