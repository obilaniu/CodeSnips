# -*- coding: utf-8 -*-


"""
YellowFin optimizer implementation
"""

"""
\begin{align*}
	\shortintertext{\textbf{SGD}}
	v_{t+1} &= \mu_{t}v_{t} + \alpha_{t} \nabla f(\theta_{t}) \\
	\theta_{t+1} &= \theta_{t} + v_{t+1}
	\shortintertext{\textbf{Nesterov Momentum}}
	v_{t+1} &= \mu_{t}v_{t} + \alpha_{t} \nabla f(\theta_{t} + \mu_{t}v_{t}) \\
	\theta_{t+1} &= \theta_{t} + v_{t+1}
	\shortintertext{\textbf{Time-Shifted Nesterov Momentum}}
	\shortintertext{\textit{Substitute}}
	\theta'_{t} &= \theta_{t} + \mu_{t}v_{t} \\
	\theta_{t} &= \theta'_{t} - \mu_{t}v_{t} \\
	\shortintertext{\textit{in the above...}}
	v_{t+1} &= \mu_{t}v_{t} + \alpha_{t} \nabla f(\theta'_{t}) \\
	\theta'_{t+1} - \mu_{t+1}v_{t+1} &= \theta'_{t} - \mu_{t}v_{t} + v_{t+1} \\
	\theta'_{t+1} &= \theta'_{t} + \mu_{t+1}v_{t+1} - \mu_{t}v_{t} + v_{t+1} \\
	\theta'_{t+1} &= \theta'_{t} + \mu_{t+1}v_{t+1} - \mu_{t}v_{t} + \mu_{t}v_{t} + \alpha_{t} \nabla f(\theta'_{t}) \\
	\theta'_{t+1} &= \theta'_{t} + \mu_{t+1}v_{t+1} - \cancel{\mu_{t}v_{t}} + \cancel{\mu_{t}v_{t}} + \alpha_{t} \nabla f(\theta'_{t}) \\
	\theta'_{t+1} &= \theta'_{t} + \mu_{t+1}v_{t+1} + \alpha_{t} \nabla f(\theta'_{t}) \\
	\shortintertext{\textit{... to get}}
	v_{t+1} &= \mu_{t}v_{t} + \alpha_{t} \nabla f(\theta'_{t}) \\
	\theta'_{t+1} &= \theta'_{t} + \mu_{t+1}v_{t+1} + \alpha_{t} \nabla f(\theta'_{t})
\end{align*}
"""

import numpy                                as np
import torch                                as T
import torch.autograd                       as TA
import torch.cuda                           as TC
import torch.optim                          as TO


class YellowFin(TO.Optimizer):
	def __init__(self, params, lr=1e-3, mom=0.0, beta=0.999, curvWW=20,
	             nesterov=False):
		#
		# YellowFin does not handle individual parameters adaptatively;
		# Indeed, the whole point is global tuning of learning rate and
		# momentum to suit all managed parameters reasonably well.
		#
		# Accordingly, we don't make use of param defaults, and make the
		# optimizer state essentially global (except for the velocity and
		# the gradient mean+variance EWMAs for gradient variance estimation)
		#
		
		self.alpha0       = float(lr)       # alpha_t
		self.mu0          = float(mom)      # mu_t
		self.beta         = float(beta)     # beta
		self.curvWW       = int(curvWW)     # curvature window width
		self.nesterov     = bool(nesterov)  # Nesterov-like momentum update?
		self.stepCount    = 0               # Global Step Count
		
		super(YellowFin, self).__init__(params, {})
	
	def step(self, closure=None):
		loss = None if closure is None else closure()
		
		#
		#     Gradient Analysis
		#
		# Collect gradient-vector statistics from each parameter, and update
		# each parameter's EWMA and variance EWMA.
		#
		
		gL2Sq = 0
		C     = 0
		for group in self.param_groups:
			for param in group["params"]:
				if param.grad is None: continue
				
				# Retrieve gradient
				g = param.grad.data
				
				
				# Ensure Per-Parameter State is initialized
				S = self.state[param]
				if "g2EWMA" not in S:   S["g2EWMA"] = T.zeros_like(g)
				if "gEWMA"  not in S:   S["gEWMA"]  = T.zeros_like(g)
				if "v"      not in S:   S["v"]      = T.zeros_like(g)
				
				
				# Algorithm 3: Gradient Variance
				g2 = g*g
				S["g2EWMA"].lerp_(g2, 1-self.beta)
				S["gEWMA"] .lerp_(g,  1-self.beta)
				C += self.allsum(T.abs(T.addcmul(S["g2EWMA"], -1.0, S["gEWMA"], S["gEWMA"])))
				
				
				# Algorithm 2: Curvature Range
				# Algorithm 4: Distance to Optimum
				gL2Sq += self.allsum(g2)
		
		#
		# Having updated the gradient/variance EWMAs of all parameters in
		# gEWMA & g2EWMA and collected the gradient's squared magnitude into
		# and its variance into C, update YellowFin's state tracker.
		#
		#     Ensure Existence of Optimizer State
		if "alpha"        not in self.__dict__: self.alpha        = T.zeros_like(gL2Sq) + self.alpha0
		if "mu"           not in self.__dict__: self.mu           = T.zeros_like(gL2Sq) + self.mu0
		if "gL2SqMinEWMA" not in self.__dict__: self.gL2SqMinEWMA = T.ones_like(gL2Sq)
		if "gL2SqMaxEWMA" not in self.__dict__: self.gL2SqMaxEWMA = T.ones_like(gL2Sq)
		if "gL2EWMA"      not in self.__dict__: self.gL2EWMA      = T.zeros_like(gL2Sq)
		if "gL2SqEWMA"    not in self.__dict__: self.gL2SqEWMA    = T.zeros_like(gL2Sq)
		if "DEWMA"        not in self.__dict__: self.DEWMA        = T.zeros_like(gL2Sq)
		#
		#     Algorithm 2: Curvature Range
		#
		# Begins with insertion of gL2Sq into ring buffer, then performs EWMA
		# update of that ring buffer's min/max.
		#
		if "gL2SqRB"      not in self.__dict__:
			self.gL2SqRB = T.zeros_like(gL2Sq) + gL2Sq
		elif len(self.gL2SqRB) < self.curvWW:
			self.gL2SqRB = T.cat([self.gL2SqRB, gL2Sq])
		else:
			insertPoint = self.stepCount % self.curvWW
			self.gL2SqRB[insertPoint:insertPoint+1] = gL2Sq
		self.gL2SqMinEWMA.lerp_(self.gL2SqRB.min(0)[0],       1-self.beta)
		self.gL2SqMaxEWMA.lerp_(self.gL2SqRB.max(0)[0],       1-self.beta)
		#
		#     Algorithm 4: Distance to Optimum
		#
		# Involves simply the update of gL2EWMA and gL2SqEWMA, followed by the
		# update of DEWMA (which uses both of them).
		#
		self.gL2EWMA     .lerp_(gL2Sq.sqrt(),                 1-self.beta)
		self.gL2SqEWMA   .lerp_(gL2Sq,                        1-self.beta)
		self.DEWMA       .lerp_(self.gL2EWMA/self.gL2SqEWMA,  1-self.beta)
		#
		#     SingleStep
		#
		# Uses C, DEWMA, gL2SqMinEWMA, gL2SqMaxEWMA to compute good alpha and
		# mu using a cubic equation.
		#
		# We seek to minimize
		#
		#     mu*D*D + alpha*alpha*C
		#
		# where
		#              sqrt(gL2SqMax/gL2SqMin) - 1
		#       mu >= (---------------------------) ** 2
		#              sqrt(gL2SqMax/gL2SqMin) + 1
		#
		#       alpha = (1 - sqrt(mu))**2 / gL2SqMin
		#
		# The paper proposes to substitute x = sqrt(mu) into the equations.
		# After some magic, this happens in TensorFlow:
		#
		p         = self.DEWMA**2 * self.gL2SqMinEWMA**2 * 0.5 / C
		w3        = -0.5*(p + T.sqrt(p**2 + 4.0/27.0*p**3))
		w         = w3.sign() * w3.abs().pow(1.0/3.0)
		root      = w - p/3/w + 1
		sqrtdr    = T.sqrt(self.gL2SqMaxEWMA/self.gL2SqMinEWMA)
		muUpd     = T.max(root**2, ((sqrtdr-1)/(sqrtdr+1))**2)
		alphaUpd  = (1 - muUpd.sqrt())**2 / self.gL2SqMinEWMA
		#
		#     Parameter Update
		#
		# This uses the current alpha/mu and the next as well.
		#
		for group in self.param_groups:
			for param in group["params"]:
				if param.grad is None: continue
				S = self.state[param]
				
				g = param.grad.data
				p = param.data
				v = S["v"]
				
				v.mul_(self.mu).addcmul_(-1.0, self.alpha, g)
				self.mu   .lerp_(muUpd,    1-self.beta)
				if self.nesterov:
					p.addcmul_(1.0, self.mu, v).addcmul_(-1.0, self.alpha, g)
				else:
					p.add_(v)
				self.alpha.lerp_(alphaUpd, 1-self.beta)
		
		#
		# Step Counter Increment
		#
		# (necessary for correct functioning of ring buffer)
		#
		self.stepCount += 1
		
		
		# Return loss
		return loss
	
	@classmethod
	def allsum(kls, x):
		x = x.squeeze()
		while x.numel() > 1:
			x = x.sum(0)
		return x
