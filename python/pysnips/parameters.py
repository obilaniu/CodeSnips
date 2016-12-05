#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Imports
import numpy                                as np



#
# Ontology:
#
# ParamSpec
#   - Belongs to a ParamSpecSet
#   - Has a unique name (string) within all ParamSpecSets it belongs to.
#   - Does **not** contain a value or instance of a parameter. Instead a
#     ParamSpec embodies a specification of a parameter, including e.g.
#       - A role (weight? bias? batch-norm related?)
#       - A value initializer.
#       - A description
#
# ParamSpecSet
#   - Is a set of ParamSpec's.
#   - Must exist outside of any function or optimizer.
#   - Also does not contain any parameter value.
#   - Serves as a summary of the parameters present in a machine learning
#     problem
#   - Offers iterators over the parameter specs.
#
# Param
#   - Is a parameter value in any form associated with a ParamSpec.
#   - Its name is that of the ParamSpec.
#   - Has a unique name (string) within all ParamSets it belongs to.
#   - Has an asNumpy() API.
#   - Does not refer to its owning ParamSet.
#
# ParamSet
#   - Is a set of Param's.
#   - Must exist without reference to any function or optimizer.
#   - Offers iterators over the parameters.
#   - Has subsetting operators.
#
# OptimizerState
#   - Is a set of Param's associated with their auxiliary data
#   - Has exactly one reference to the Optimizer that created it, and to no other.
#   - Does not refer to the function that owns the optimizer.
#
# Optimizer
#   - Defined in PySnips.optimizers
#   - Embodies the logic for:
#       - Initializing the optimizer
#       - Making a pre-update  step
#       - Making a post-update step given the gradient
#   - Does NOT store any OptimizerState or parameter value.
#
# Function
#   - Embodies a function that depends on a ParamSet.
#   - May use multiple Optimizer's for different variables.
#   - Shall not use more than one Optimizer for the same variable.
#



#
# Initializers
#


#
# Base Parameter Specification Class
#

class ParamSpec(object):
	def __init__(self, name, shape, dtype, init="zero", gain=1.0, **kwargs):
		shape          = tuple(shape)
		name           = isinstance(name, str) and name
		if not name:
			raise ValueError()
		
		self.__dict__ += kwargs
		self.name      = name
		self.shape     = shape
		self.dtype     = dtype
		self.gain      = gain
		
		if   callable(init):
			self.initFn = init
		elif isinstance(init, str):
			self.initFn = {}[init]
		else:
			raise TypeError()
	def getName(self):
		return self.name
	def getShape(self):
		return self.shape
	def getDtype(self):
		return self.dtype
	def getNumParams(self):
		return reduce(lambda a,b: a*b, self.getShape(), 1)
	def init(self):
		if   callable(self.initFn):
			return self.initFn(self)
		else:
			return np.zeros(self.getShape(), self.getDtype())
			

# Parameter Roles
class ParamBias(ParamSpec):
	pass
class ParamWeight(ParamSpec):
	pass

# Specialized Parameter Classes
class ConvWeight(ParamWeight):
	pass
class BNWeight(ParamWeight):   # BN Gamma parameter
	pass
class BNBias(ParamBias):       # BN Beta  parameter
	pass
class BNMean(ParamBias):       # BN Mu    parameter
	pass
class BNStd(ParamWeight):      # BN Sigma parameter
	pass

# Parameter Collection
class ParamSet(object):
	def __init__(self):
		self.p = {}
	def __getitem__(self, key):
		pass
	def __setitem__(self, key, value):
		pass
	def __delitem__(self, key):
		pass
	def add(self, p):
		if isinstance(p, Param):
			if p.name in self.p:
				raise KeyError()
			else:
				self.p[p.getName()] = p
		else:
			raise TypeError()
	def getNumParams(self):
		return reduce(lambda a,b: a+b.getNumParams(), self.p.values(), 0)


#
# OptimizerState
#

class OptimizerState(object):
	def __init__(self, optimizer, paramSet):
		self.optimizer    = optimizer
		self.paramSet     = paramSet
		self.currentState = {}
		self.nextState    = None
	
	def dump(self):
		pass
	def load(self):
		pass
	
	def pre(self):
		pass
	def post(self):
		pass


