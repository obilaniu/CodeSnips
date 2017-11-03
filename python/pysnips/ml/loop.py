# -*- coding: utf-8 -*-

# Imports
import numpy as np, os, sys, time, yaml as Y



#
# Callbacks manipulate a single state dictionary. This dictionary contains keys
# in a pseudo-filesystem format. The to-level "std" namespace is reserved for the
# implementation and contains the following keys:
#
#     std/loop/state    (required)
#     std/loop/epochNum (required)
#     std/loop/batchNum (required)
#     std/loop/stepNum  (required)
#     std/loop/epochMax (optional)
#     std/loop/batchMax (optional)
#
# Users should prefer adding keys to the namespace
#
#     user/*
#

class Callback(object):
	def anteTrain(self, d): pass
	def anteEpoch(self, d): pass
	def anteBatch(self, d): pass
	def execBatch(self, d): pass
	def postBatch(self, d): pass
	def postEpoch(self, d): pass
	def postTrain(self, d): pass
	def finiTrain(self, d): pass
	def finiEpoch(self, d): pass
	def finiBatch(self, d): pass
	def preempt  (self, d): pass




class CallbackLambda(Callback):
	def __init__(self,
	             anteTrain = lambda self,d: None,
	             anteEpoch = lambda self,d: None,
	             anteBatch = lambda self,d: None,
	             execBatch = lambda self,d: None,
	             postBatch = lambda self,d: None,
	             postEpoch = lambda self,d: None,
	             postTrain = lambda self,d: None,
	             finiTrain = lambda self,d: None,
	             finiEpoch = lambda self,d: None,
	             finiBatch = lambda self,d: None,
	             preempt   = lambda self,d: None,
	             **kwargs):
		kwargs = {k:v for k,v in kwargs.iteritems() if not k.startswith("__")}
		self.__dict__.update(kwargs)
		self.__anteTrain = anteTrain
		self.__anteEpoch = anteEpoch
		self.__anteBatch = anteBatch
		self.__execBatch = execBatch
		self.__postBatch = postBatch
		self.__postEpoch = postEpoch
		self.__postTrain = postTrain
		self.__finiTrain = finiTrain
		self.__finiEpoch = finiEpoch
		self.__finiBatch = finiBatch
		self.__preempt   = preempt
	def anteTrain(self, d): self.__anteTrain(self, d)
	def anteEpoch(self, d): self.__anteEpoch(self, d)
	def anteBatch(self, d): self.__anteBatch(self, d)
	def execBatch(self, d): self.__execBatch(self, d)
	def postBatch(self, d): self.__postBatch(self, d)
	def postEpoch(self, d): self.__postEpoch(self, d)
	def postTrain(self, d): self.__postTrain(self, d)
	def finiTrain(self, d): self.__finiTrain(self, d)
	def finiEpoch(self, d): self.__finiEpoch(self, d)
	def finiBatch(self, d): self.__finiBatch(self, d)
	def preempt  (self, d): self.__preempt  (self, d)




class CallbackSetter(Callback):
	def __init__(self,
	             anteTrain = {},
	             anteEpoch = {},
	             anteBatch = {},
	             execBatch = {},
	             postBatch = {},
	             postEpoch = {},
	             postTrain = {},
	             finiTrain = {},
	             finiEpoch = {},
	             finiBatch = {},
	             preempt   = {}):
		self.__anteTrain = anteTrain
		self.__anteEpoch = anteEpoch
		self.__anteBatch = anteBatch
		self.__execBatch = execBatch
		self.__postBatch = postBatch
		self.__postEpoch = postEpoch
		self.__postTrain = postTrain
		self.__finiTrain = finiTrain
		self.__finiEpoch = finiEpoch
		self.__finiBatch = finiBatch
		self.__preempt   = preempt
	def anteTrain(self, d): d.update(self.__anteTrain)
	def anteEpoch(self, d): d.update(self.__anteEpoch)
	def anteBatch(self, d): d.update(self.__anteBatch)
	def execBatch(self, d): d.update(self.__execBatch)
	def postBatch(self, d): d.update(self.__postBatch)
	def postEpoch(self, d): d.update(self.__postEpoch)
	def postTrain(self, d): d.update(self.__postTrain)
	def finiTrain(self, d): d.update(self.__finiTrain)
	def finiEpoch(self, d): d.update(self.__finiEpoch)
	def finiBatch(self, d): d.update(self.__finiBatch)
	def preempt  (self, d): d.update(self.__preempt)




class CallbackList(Callback):
	def __init__(self, cblist):
		self.cblist = cblist
	def anteTrain(self, d):
		for cb in self.cblist[::+1]: cb.anteTrain(d)
	def anteEpoch(self, d):
		for cb in self.cblist[::+1]: cb.anteEpoch(d)
	def anteBatch(self, d):
		for cb in self.cblist[::+1]: cb.anteBatch(d)
	def execBatch(self, d):
		for cb in self.cblist[::+1]: cb.execBatch(d)
	def postBatch(self, d):
		for cb in self.cblist[::-1]: cb.postBatch(d)
	def postEpoch(self, d):
		for cb in self.cblist[::-1]: cb.postEpoch(d)
	def postTrain(self, d):
		for cb in self.cblist[::-1]: cb.postTrain(d)
	def finiTrain(self, d):
		for cb in self.cblist[::+1]: cb.finiTrain(d)
	def finiEpoch(self, d):
		for cb in self.cblist[::+1]: cb.finiEpoch(d)
	def finiBatch(self, d):
		for cb in self.cblist[::+1]: cb.finiBatch(d)
	def preempt  (self, d):
		for cb in self.cblist[::+1]: cb.preempt  (d)
	




class CallbackProgbar(Callback):
	def __init__(self, barLen, action=sys.stdout):
		self.barLen = barLen
		if   callable(action):
			self.action = action
		elif hasattr(action, "write"):
			self.action = lambda s: action.write(s)
		else:
			raise ValueError("Must provide unary callable or object with write(arg) method, such as a stream!")
	def finiBatch(self, d):
		epochNum = d["std/loop/epochNum"]
		batchNum = d["std/loop/batchNum"]
		stepNum  = d["std/loop/stepNum"]
		batchMax = d.get("std/loop/batchMax", 0)
		self.action(epochprogbar(self.barLen,
		                         epochNum,
		                         batchNum+1,  # +1 because 0-based.
		                         stepNum+1,   # +1 because 0-based.
		                         batchMax))



class CallbackFlush(Callback):
	def __init__(self, batchFlush=True, epochFlush=True, action=sys.stdout):
		self.batchFlush = bool(batchFlush)
		self.epochFlush = bool(epochFlush)
		if   callable(action):
			self.action = action
		elif hasattr(action, "flush"):
			self.action = lambda: action.flush()
		else:
			raise ValueError("Must provide nullary callable or object with flush() method, such as a stream!")
	def finiEpoch(self, d):
		if self.epochFlush: self.action()
	def finiBatch(self, d):
		if self.batchFlush: self.action()



class CallbackLinefeed(Callback):
	def __init__(self,
	             batchPrint = "\r",
	             epochPrint = os.linesep,
	             action     = sys.stdout):
		self.batchPrint = "" if batchPrint is None else batchPrint
		self.epochPrint = "" if epochPrint is None else epochPrint
		if   callable(action):
			self.action = action
		elif hasattr(action, "write"):
			self.action = lambda s: action.write(s)
		else:
			raise ValueError("Must provide unary callable or object with write(arg) method, such as a stream!")
	def finiEpoch(self, d):
		self.action(self.epochPrint)
	def finiBatch(self, d):
		self.action(self.batchPrint)



#
# FUNCTIONS
#

#
# Master training loop
#

def loop(callbacks, d={}):
	d["std/loop/state"   ] = d.get("std/loop/state",    "anteTrain")
	d["std/loop/batchNum"] = d.get("std/loop/batchNum", 0)
	d["std/loop/epochNum"] = d.get("std/loop/epochNum", 0)
	d["std/loop/stepNum" ] = d.get("std/loop/stepNum",  0)
	
	
	if not isinstance(callbacks, (Callback, list)):
		raise TypeError("cbs must be a callback or list of callbacks!")
	if not isinstance(callbacks, list):
		callbacks = [callbacks]
	if not isinstance(callbacks, CallbackList):
		callbacks = CallbackList(callbacks)
	
	
	def getState():
		return d["std/loop/state"]
	def setState(state):
		d["std/loop/state"]     = state
	def updBatch():
		d["std/loop/batchNum"] += 1
		d["std/loop/stepNum" ] += 1
	def updEpoch():
		d["std/loop/epochNum"] += 1
		d["std/loop/batchNum"]  = 0
	def attemptEpoch():
		epochMax = d.get("std/loop/epochMax", None)
		return epochMax is None or d["std/loop/epochNum"] < epochMax
	def attemptBatch():
		batchMax = d.get("std/loop/batchMax", None)
		return batchMax is None or d["std/loop/batchNum"] < batchMax
	
	
	while True:
		if   getState() in ["anteTrain", None]:
			pass;   callbacks.anteTrain(d);               setState("anteEpoch");
		elif getState() in ["anteEpoch"]:
			try:
				if attemptEpoch():
					callbacks.anteEpoch(d);               setState("anteBatch");
				else:
					raise StopIteration()
			except  StopIteration as si:                  setState("postTrain");
		elif getState() in ["anteBatch"]:
			try:
				if attemptBatch():
					callbacks.anteBatch(d);               setState("execBatch");
				else:
					raise StopIteration()
			except  StopIteration as si:                  setState("postEpoch");
		elif getState() in ["execBatch"]:
			pass;   callbacks.execBatch(d);               setState("postBatch");
		elif getState() in ["postBatch"]:
			pass;   callbacks.postBatch(d);               setState("finiBatch");
		elif getState() in ["finiBatch"]:
			pass;   callbacks.finiBatch(d);  updBatch();  setState("anteBatch");
		elif getState() in ["postEpoch"]:
			pass;   callbacks.postEpoch(d);               setState("finiEpoch");
		elif getState() in ["finiEpoch"]:
			pass;   callbacks.finiEpoch(d);  updEpoch();  setState("anteEpoch");
		elif getState() in ["postTrain"]:
			pass;   callbacks.postTrain(d);               setState("finiTrain");
		elif getState() in ["finiTrain"]:
			pass;   callbacks.finiTrain(d);               setState("doneTrain");
		elif getState() in ["doneTrain"]:
			break
		
		callbacks.preempt(d)
	
	return d



#
# Progress bar synthesis
#


def progbar(barLen, barFrac, delim=True):
	if   isinstance(barFrac, float):
		barFrac  = max(0.0, min(1.0,    barFrac))
		a        = int(barFrac*barLen)
	elif isinstance(barFrac, (int, long)):
		a        = max(0,   min(barLen, barFrac))
	else:
		raise InvalidArgument("barFrac must be a floating-point number in the range [0.0, 1.0] "\
		                      " or an integer in the range [0, barLen]!")
	
	b = barLen-a
	
	if   b == 0: s = "="*barLen
	elif a == 0: s = ">" + " "*(barLen-1)
	else:        s = "="*(a-1) + ">" + " "*b
	return       "["+s+"]" if delim else s

def epochprogbar(l, e, b, s, Nb, delim=True):
	if   Nb <= 0:
		f   = 1.0 - abs(float(b % (2*l))/l - 1.0)  # Oscillate
		fs  = "{:07d}".format(b)
		sep = ":"
	else:
		f   = max(0.0, min(1.0, float(b)/Nb))      # True fraction
		fs  = "{:.7f}".format(f)
		fs  = "DONE   " if fs.startswith("1") else fs[-7:]
		sep = "."
	
	bar = progbar(l, f, delim)
	
	return "Epoch {e:4d}{sep:s}{fs:s}  {bar:s}".format(**locals())


#
# YAML Ser/Des utilities
#

def dumpStdLoopState(data, **kwargs):
	loopState = {k: data.get(k, None) for k in [
		"std/loop/state",
		"std/loop/batchNum",
		"std/loop/epochNum",
		"std/loop/stepNum",
		"std/loop/batchMax",
		"std/loop/epochMax",
	]}
	return Y.dump(loopState, **kwargs)
def loadStdLoopState(stream, **kwargs):
	loopState = Y.load(stream, **kwargs)
	loopState = {k: loopState.get(k, None) for k in [
		"std/loop/state",
		"std/loop/batchNum",
		"std/loop/epochNum",
		"std/loop/stepNum",
		"std/loop/batchMax",
		"std/loop/epochMax",
	]}
	return loopState
def dumpNumpyPRNGState(data, **kwargs):
	name, state, position, hasGaussian, cachedGaussian = data
	prngState = {
		"name":           name,
		"state":          map(int, state.tolist()),
		"position":       position,
		"hasGaussian":    hasGaussian,
		"cachedGaussian": cachedGaussian,
	}
	return Y.dump(prngState, **kwargs)
def loadNumpyPRNGState(stream, **kwargs):
	prngState = Y.load(stream, **kwargs)
	return (
		prngState["name"],
		np.array(prngState["state"], np.uint32),
		prngState["position"],
		prngState["hasGaussian"],
		prngState["cachedGaussian"],
	)




# Short testcase
if __name__ == "__main__":
	slp  = CallbackLambda(execBatch=lambda self,d:time.sleep(1e-2))
	prog = CallbackProgbar(50)
	lf   = CallbackLinefeed()
	fl   = CallbackFlush()
	
	loop([slp, prog, lf, fl],
	     {"std/loop/epochMax": 200,
	      "std/loop/batchMax": 150})



