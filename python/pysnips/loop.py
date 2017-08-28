# -*- coding: utf-8 -*-

# Imports
import os, sys, time



#
# Callbacks manipulate a single state dictionary. This dictionary contains keys
# in a reverse.DNS format. The to-level "std" namespace is reserved for the
# implementation and contains the following keys:
#
#     std.loop.epochNum (required)
#     std.loop.batchNum (required)
#     std.loop.epochMax (optional)
#     std.loop.batchMax (optional)
#
# Users should prefer adding keys to the namespace
#
#     user.*
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
	             finiBatch = {}):
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




class CallbackTiming(Callback):
	def anteTrain(self, d): d["std.timing.anteTrain"] = time.time()
	def anteEpoch(self, d): d["std.timing.anteEpoch"] = time.time()
	def anteBatch(self, d): d["std.timing.anteBatch"] = time.time()




class CallbackProgbar(Callback):
	def __init__(self, barLen, stream=sys.stdout):
		self.barLen     = barLen
		self.stream     = stream
	def finiBatch(self, d):
		epochNum = d["std.loop.epochNum"]
		batchNum = d["std.loop.batchNum"]
		batchMax = d.get("std.loop.batchMax", 0)
		self.stream.write(epochprogbar(self.barLen,
		                               epochNum,
		                               batchNum+1,  # +1 because 0-based.
		                               batchMax))



class CallbackLinefeed(Callback):
	def __init__(self,
	             batchPrint = "\r",
	             epochPrint = os.linesep,
	             batchFlush = True,
	             epochFlush = True,
	             stream     = sys.stdout):
		self.batchPrint = "" if batchPrint is None else batchPrint
		self.epochPrint = "" if epochPrint is None else epochPrint
		self.batchFlush = bool(batchFlush)
		self.epochFlush = bool(epochFlush)
		self.stream     = stream
	def finiEpoch(self, d):
		self.stream.write(self.epochPrint)
		if self.epochFlush:
			self.stream.flush()
	def finiBatch(self, d):
		self.stream.write(self.batchPrint)
		if self.batchFlush:
			self.stream.flush()



class CallbackSnapshot(Callback):
	def __init__(self, expmt):
		self.expmt = [expmt] if not isinstance(expmt, list) else expmt
	def finiTrain(self, d):
		for e in self.expmt:
			e.snapshot()




def loop(cbs, userdict={}):
	if not isinstance(cbs, (Callback, list)):
		raise TypeError("cbs must be a callback or list of callbacks!")
	if not isinstance(cbs, list):
		cbs  = [cbs]
	cbs = CallbackList(cbs)
	
	
	
	def continueTrain(d):
		return "std.loop.epochMax" not in d or \
		       d["std.loop.epochNum"] < d["std.loop.epochMax"]
	def continueEpoch(d):
		return "std.loop.batchMax" not in d or \
		       d["std.loop.batchNum"] < d["std.loop.batchMax"]
	dictTrain = {"std.loop.epochNum": 0,
	             "std.loop.batchNum": 0}
	dictTrain.update(userdict)
	
	
	
	cbs.anteTrain(dictTrain)
	while continueTrain(dictTrain):
		dictEpoch = dictTrain.copy()
		
		try:   cbs.anteEpoch(dictEpoch)
		except StopIteration as si: break
		
		while continueEpoch(dictEpoch):
			dictBatch = dictEpoch.copy()
			
			try:   cbs.anteBatch(dictBatch)
			except StopIteration as si: break
			
			try:
				cbs.execBatch(dictBatch)
				cbs.postBatch(dictBatch)
				cbs.finiBatch(dictBatch)
			except StopIteration as si:
				print("A callback's execBatch(), postBatch() or finiBatch() functions illegally raised an exception!")
				raise si
			finally:
				dictEpoch["std.loop.batchNum"] += 1
		try:
			cbs.postEpoch(dictEpoch)
			cbs.finiEpoch(dictEpoch)
		except StopIteration as si:
			print("A callback's postEpoch() or finiEpoch() functions illegally raised an exception!")
			raise si
		finally:
			dictTrain["std.loop.batchNum"]  = 0
			dictTrain["std.loop.epochNum"] += 1
	cbs.postTrain(dictTrain)
	cbs.finiTrain(dictTrain)



def progbar(barLen, barFrac, delim=True):
	if   isinstance(barFrac, float):
		barFrac  = max(0.0, min(1.0,    barFrac))
		a        = int(barFrac*barLen)
	elif isinstance(barFrac, (int, long)):
		a        = max(0,   min(barLen, barFrac))
	else:
		raise InvalidArgument("barFrac must be a floating-point number in the range [0.0, 1.0] "\
		                      " or an integer in the range [0, barLen]!")
	b  = barLen-a
	
	s  = u"\u2588"*a + u" "*b
	if delim:
		s = u"\u23B9" + s + u"\u23B8"
	return s

def epochprogbar(barLen, epochNum, batchNum, numBatches, delim=True):
	if   numBatches <= 0:
		barFrac  = 1.0 - abs(float(batchNum % (2*barLen))/barLen - 1.0)  # Oscillate
		barSep   = u"+"
		strFrac  = u"{:07d}".format(batchNum)
	else:
		barFrac  = max(0.0, min(1.0, float(batchNum)/numBatches))        # True fraction
		barSep   = u"."
		strFrac  = u"{:.7f}".format(barFrac)
		if strFrac.startswith(u"1"): strFrac  = u"DONE   "
		else:                        strFrac  = strFrac[-7:]
	s        = u""
	s       += u"Epoch {:4d}{:s}{:s}".format(epochNum, barSep, strFrac)
	s       += progbar(barLen, barFrac, delim)
	
	return s


#
#if __name__ == "__main__":
#	slp  = CallbackLambda(execBatch=lambda self,d:time.sleep(0.01))
#	prog = CallbackProgbar(50)
#	lf   = CallbackLinefeed()
#	
#	loop([slp, prog, lf],
#	     {"std.loop.epochMax": 200,
#	      "std.loop.batchMax": 150})
#


