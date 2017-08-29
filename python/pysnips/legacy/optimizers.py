#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Imports
import numpy                                as np


#
# Optimizer Value Base Class
#

class OptimizerValue(object):
	def __init__(self, name, **kwargs):
		self.__dict__.update(kwargs)
		self.name = name
	def getName(self):
		return self.name
class StateValue(OptimizerValue):
	def __init__(self, name, **kwargs):
		super(StateValue, self).__init__(name, **kwargs)
class LocalStateValue(StateValue):
	def __init__(self, name, value=0., dtype=None, **kwargs):
		self.value = value
		self.dtype = dtype
		super(LocalStateValue, self).__init__(name, **kwargs)
	def init(self, p):
		return np.full_like(p, self.value, self.dtype)
class ParameterValue(LocalStateValue):
	def __init__(self, name="p", value=0., dtype=None, **kwargs):
		super(ParameterValue, self).__init__(name, value, dtype, **kwargs)
	def init(self, p):
		return p
class MomentValue(LocalStateValue):
	pass
class GlobalStateValue(StateValue):
	def __init__(self, name, value=0., **kwargs):
		self.value = value
		super(GlobalStateValue, self).__init__(name, **kwargs)
	def init(self):
		return self.value
class HyperparameterValue(OptimizerValue):
	def __init__(self, name, defaultValue, **kwargs):
		self.defaultValue = defaultValue
		super(HyperparameterValue, self).__init__(name, **kwargs)
	def getDefault(self):
		return self.defaultValue
class LearningRateValue(HyperparameterValue):
	def __init__(self, name="alpha", defaultValue=0.001, **kwargs):
		super(LearningRateValue, self).__init__(name, defaultValue, **kwargs)



#
# Optimizer Base Class.
#

class Optimizer(object):
	def __init__(self, *args, **kwargs):
		self.setupInterface(*args, **kwargs)
		self.checkInterface()
		self.setupDict()
	
	def setupInterface(self, *args, **kwargs):
		self.lS    = [ParameterValue()]
		self.gS    = []
		self.hyper = []
	
	def checkInterface(self):
		# Three lists...
		assert(isinstance(self.lS,    list))
		assert(isinstance(self.gS,    list))
		assert(isinstance(self.hyper, list))
		# Containing appropriate OptimizationValues...
		for v in self.lS:    assert(isinstance(v, LocalStateValue))
		for v in self.gS:    assert(isinstance(v, GlobalStateValue))
		for v in self.hyper: assert(isinstance(v, HyperparameterValue))
		# First local state is always the parameter...
		assert(isinstance(self.lS[0], ParameterValue))
		# And it must appear exactly once. The learning rate can appear at most once.
		assert(sum([isinstance(v, ParameterValue)    for v in self.lS   ]) == 1)
		assert(sum([isinstance(v, LearningRateValue) for v in self.hyper]) <= 1)
	
	def setupDict(self):
		self.interfaceDict = {}
		for v in self.lS+self.gS+self.hyper:
			self.interfaceDict[v.getName()] = v
	
	def lPre (self,       lS, gS, **kwargs):
		"""Pre-update  local  optimizer states."""
		return lS
	
	def lPost(self, grad, lS, gS, **kwargs):
		"""Post-update local  optimizer states given the current gradient."""
		return lS
	
	def gPre (self,       gS, **kwargs):
		"""Pre-update  global optimizer states."""
		return gS
	
	def gPost(self,       gS, **kwargs):
		"""Post-update global optimizer states."""
		return gS
	
	def lInit(self, p):
		"""Initialize the local optimizer state, given a Numpy
		representation of the parameter.
		"""
		return tuple([v.init(p) for v in self.lS])
	
	def gInit(self):
		"""Initialize the global optimizer state."""
		return tuple([v.init()  for v in self.gS])
	
	def getLSNames(self):
		"""Get local states names."""
		return [v.getName() for v in self.lS]
	
	def getGSNames(self):
		"""Get global states names."""
		return [v.getName() for v in self.gS]
	
	def getHyperNames(self):
		"""Get hyperparameter names."""
		return [v.getName() for v in self.hyper]
	
	def getNumHypers(self):
		"""Get number of hyperparameter arguments."""
		return len(self.hyper)
	
	def getHyperDefaults(self):
		"""Get dictionary of hyperparameter defaults."""
		return {v.getName():v.getDefault() for v in self.hyper}
	
	def interpretHypers(self, *args, **kwargs):
		v = args[:min(len(args), self.getNumHypers())]
		
		for i in xrange(len(v), self.getNumHypers()):
			h  = self.hyper[i]
			v += (kwargs.get(h.getName(), h.getDefault()),)
		
		return v

#
# SGD
#
# URL: prehistoric
#
# grad  is the gradient
# p     is the parameter
# m     is the 1st-order moment  (init: 0.0)
#

class SGD(Optimizer):
	def setupInterface(self, *args, **kwargs):
		self.lS    = [ParameterValue     ("p"),
		              MomentValue        ("m")]
		self.gS    = []
		self.hyper = [LearningRateValue  ("alpha",    kwargs.get("alpha",    0.001)),
		              HyperparameterValue("mom",      kwargs.get("mom",      0.9)),
		              HyperparameterValue("nesterov", kwargs.get("nesterov", False))]
	
	def lPost(self, grad, (p, m), gS, *args, **kwargs):
		alpha, mom, nesterov = self.interpretHypers(*args, **kwargs)
		
		ag = alpha*grad
		
		if nesterov:
			m  = (mom)*m - ag
			p += (mom)*m - ag
		else:
			                   # Also equivalent to:
			m  = (mom)*m - ag  #     p += (mom)*m - ag
			p +=       m       #     m  = (mom)*m - ag
			                   # Except here we did CSE (common subexpression
			                   # elimination)
		
		return (p, m)

class PlainSGD(SGD):
	def setupInterface(self, *args, **kwargs):
		kwargs["nesterov"] = False
		super(PlainSGD, self).setupInterface(*args, **kwargs)

class NAG     (SGD):
	def setupInterface(self, *args, **kwargs):
		kwargs["nesterov"] = True
		super(NAG,      self).setupInterface(*args, **kwargs)


#
# Adam
#
# URL: https://arxiv.org/abs/1412.6980
#
# grad  is the gradient
# p     is the parameter
# m     is the 1st-order moment  (init: 0.0)
# v     is the 2nd-order moment  (init: 0.0)
# t     is an integer counter    (init: 1)
# alpha is the learning rate
#
# Returns the new p,m,v values.
#

class Adam(Optimizer):
	def setupInterface(self, *args, **kwargs):
		self.lS    = [ParameterValue     ("p"),
		              MomentValue        ("m"),
		              MomentValue        ("v")]
		self.gS    = [GlobalStateValue   ("t",        np.array(1, np.int64))]
		self.hyper = [LearningRateValue  ("alpha",    kwargs.get("alpha",    0.001)),
		              HyperparameterValue("beta1",    kwargs.get("beta1",    0.9)),
		              HyperparameterValue("beta2",    kwargs.get("beta2",    0.999)),
		              HyperparameterValue("eps",      kwargs.get("eps",      1e-8))]
	def lPost(self, grad, (p, m, v), (t,), *args, **kwargs):
		alpha, beta1, beta2, eps = self.interpretHypers(*args, **kwargs)
		
		m  = (beta1)*m + (1-beta1)*(grad   )
		v  = (beta2)*v + (1-beta2)*(grad**2)
		m /= (1 - beta1**t)
		v /= (1 - beta2**t)
		p -= alpha*m/(v**0.5 + eps)
		
		return (p, m, v)
	def gPost(self, (t,), *args, **kwargs):
		return (t+1,)

#
# SMORMS3
#
# URL: http://sifter.org/~simon/journal/20150420.html
#
# grad      is the gradient
# p         is the parameter
# mem       is something wierd       (init: 1.0)
# g         is the 1st-order moment  (init: 0.0)
# g2        is the 2nd-order moment  (init: 0.0)
# alpha     is the learning rate
# minimum   is an element-wise minimum function minimum(a,b) supporting scalar
#           broadcasting.
#             For Numpy:  np.minimum
#             For Theano: theano.tensor.minimum
#
# Returns the new p,mem,g,g2 values.
#

class SMORMS3(Optimizer):
	def setupInterface(self, *args, **kwargs):
		self.lS    = [ParameterValue     ("p"),
		              LocalStateValue    ("mem",      1.0),
		              MomentValue        ("g"),
		              MomentValue        ("g2")]
		self.gS    = []
		self.hyper = [LearningRateValue  ("alpha",    kwargs.get("alpha",    0.001)),
		              HyperparameterValue("eps",      kwargs.get("eps",      1e-16)),
		              HyperparameterValue("minimum",  kwargs.get("minimum",  np.minimum))]
	def lPost(self, grad, (p, mem, g, g2), gS, *args, **kwargs):
		alpha, eps, minimum = self.interpretHypers(*args, **kwargs)
		
		r      = 1/(mem+1.)
		g      = (1-r)*g  + (r)*(grad   )
		g2     = (1-r)*g2 + (r)*(grad**2)
		g2e    = g2  + eps
		ggg2e  = g*g / g2e
		p     -= grad * minimum(alpha, ggg2e) / (g2**0.5 + eps)
		mem    = 1 + mem*(1-ggg2e)
		
		return (p, mem, g, g2)

