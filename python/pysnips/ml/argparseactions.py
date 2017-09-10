# -*- coding: utf-8 -*-

# Imports
import argparse as Ap
import ast


#
# OptimizerAction Action
#

class OptimizerAction(Ap.Action):
	def __call__(self, parser, namespace, values, option_string):
		#
		# Create new namespace
		#
		
		setattr(namespace, self.dest, Ap.Namespace())
		ns = getattr(namespace, self.dest)
		
		#
		# Split the argument string:
		#
		# --arg optimizername:key0,key1,key2=value0,key3=value1
		#
		split  = values.split(":", 1)
		name   = split[0].strip()
		rest   = split[1].split(",") if len(split) == 2 else []
		args   = []
		kwargs = {}
		
		for r in rest:
			s    = r.split("=", 1)
			s[0] = s[0].strip()
			if   len(s) == 1:
				if len(kwargs) > 0:
					raise ValueError("Positional optimizer argument \""+r+"\" found after first keyword argument!")
				args += [ast.literal_eval(s[0])]
			else:
				s[1] = s[1].strip()
				try:    kwargs[s[0]] = ast.literal_eval(s[1])
				except: kwargs[s[0]] = s[1]
		
		#
		# Parse argument string according to optimizer
		#
		if   name in ["sgd"]:
			ns.__dict__.update(OptimizerAction.filterSGD(*args, **kwargs))
		elif name in ["nag"]:
			ns.__dict__.update(OptimizerAction.filterNAG(*args, **kwargs))
		elif name in ["adam"]:
			ns.__dict__.update(OptimizerAction.filterAdam(*args, **kwargs))
		elif name in ["rmsprop"]:
			ns.__dict__.update(OptimizerAction.filterRmsprop(*args, **kwargs))
		elif name in ["yfin", "yellowfin"]:
			ns.__dict__.update(OptimizerAction.filterYellowfin(*args, **kwargs))
	
	@staticmethod
	def filterSGD      (lr=1e-3, mom=0.9, nesterov=False):
		assert isinstance(lr,       (int, long, float)) and lr    >  0
		assert isinstance(mom,      (int, long, float)) and mom   >= 0 and mom   < 1
		assert isinstance(nesterov, bool)
		
		lr    = float(lr)
		mom   = float(mom)
		
		d = locals()
		d["name"] = "sgd"
		return d
	@staticmethod
	def filterNAG      (lr=1e-3, mom=0.9, nesterov=True):
		d = OptimizerAction.filterSGD(lr, mom, nesterov)
		d["name"] = "nag"
		return d
	@staticmethod
	def filterAdam     (lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
		assert isinstance(lr,       (int, long, float)) and lr    >  0
		assert isinstance(beta1,    (int, long, float)) and beta1 >= 0 and beta1 < 1
		assert isinstance(beta2,    (int, long, float)) and beta2 >= 0 and beta2 < 1
		assert isinstance(eps,      (int, long, float)) and eps   >= 0
		
		lr    = float(lr)
		beta1 = float(beta1)
		beta2 = float(beta2)
		eps   = float(eps)
		
		d = locals()
		d["name"] = "adam"
		return d
	@staticmethod
	def filterRmsprop  (lr=1e-3, rho=0.9, eps=1e-8):
		assert isinstance(lr,       (int, long, float)) and lr    >  0
		assert isinstance(rho,      (int, long, float)) and rho   >= 0 and rho   < 1
		assert isinstance(eps,      (int, long, float)) and eps   >= 0
		
		lr    = float(lr)
		rho   = float(rho)
		eps   = float(eps)
		
		d = locals()
		d["name"] = "rmsprop"
		return d
	@staticmethod
	def filterYellowfin(lr=1.0, mom=0.0, beta=0.999, curvwindowwidth=20, nesterov=False):
		assert isinstance(lr,       (int, long, float)) and lr    >  0
		assert isinstance(mom,      (int, long, float)) and mom   >= 0 and mom   < 1
		assert isinstance(beta,     (int, long, float)) and beta  >= 0 and beta <= 1
		assert isinstance(curvwindowwidth, (int, long)) and curvwindowwidth >= 3
		assert isinstance(nesterov, bool)
		
		lr    = float(lr)
		mom   = float(mom)
		beta  = float(beta)
		curvwindowwidth = int(curvwindowwidth)
		eps   = float(eps)
		
		d = locals()
		d["name"] = "yellowfin"
		return d

