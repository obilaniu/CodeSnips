# -*- coding: utf-8 -*-

# Imports
import numpy as np, warnings


#
# Collect statistics on tensors in a dictionary.
#
#     Interesting statistics about tensors:
#
#       - Shape
#       - CountElem/Count0/CountPInf/CountNInf/CountPFin/CountNFin/CountNaN
#       - Sum
#       - L1/L2 Norm
#       - Min/Mean/Med/Max
#       - Std/Var
#       - Histogram of IEEE Std 754 sign&exponents
#

def tensorstats(tensor, prefixName=None):
	# Assert Numpy Tensor
	assert isinstance(tensor, np.ndarray)
	
	#
	# Construct histogram bins carefully.
	#
	# All the bins are [lower, upper) inclusive-exclusive except the last one,
	# which is [lower, upper] inclusive.
	#
	
	pbins = [2.0**k for k in xrange(-127,+127)]
	nbins = map(lambda x:float(np.nextafter(-x, 0)), pbins[::-1])
	bins  = [-np.inf, float(np.nextafter(-np.inf, 0))] + \
	        nbins +                                      \
	        [0.0, float(np.nextafter(0.0, 1))] +         \
	        pbins +                                      \
	        [+np.inf, +np.inf]
	
	#
	# Compute statistics while suppressing RuntimeWarnings
	#
	
	with warnings.catch_warnings():
		statShape     = map(int, tensor.shape)
		statCountElem = int  (tensor.size)
		statCount0    = int  (np.count_nonzero(tensor == 0))
		statCountPInf = int  (np.count_nonzero(tensor == +np.inf))
		statCountNInf = int  (np.count_nonzero(tensor == -np.inf))
		statCountPFin = int  (np.count_nonzero(tensor  > 0)) - statCountPInf
		statCountNFin = int  (np.count_nonzero(tensor  < 0)) - statCountNInf
		statCountNaN  = int  (np.count_nonzero(tensor != tensor))
		statSum       = float(np.nansum(tensor))
		statL1        = float(np.nansum(np.abs(tensor)))
		statL2        = float(np.nansum(np.abs(tensor)**2))
		statMin       = float(np.nanmin(tensor))
		statMean      = float(np.nanmean(tensor))
		statMedian    = float(np.nanmedian(tensor))
		statMax       = float(np.nanmax(tensor))
		statStd       = float(np.nanstd(tensor))
		statVar       = float(np.nanvar(tensor))
		statHisto     = map(int, np.histogram(tensor, bins)[0].tolist())
	
	#
	# Handle prefix in dictionary entries
	#
	
	if prefixName in [None, ""]:
		prefixName = ""
	else:
		prefixName += "" if prefixName.endswith("/") else "/"
	
	#
	# Construct and return statistics dictionary
	#
	
	return {
		prefixName+"shape":     statShape,
		prefixName+"countElem": statCountElem,
		prefixName+"count0":    statCount0,
		prefixName+"countPInf": statCountPInf,
		prefixName+"countNInf": statCountNInf,
		prefixName+"countPFin": statCountPFin,
		prefixName+"countNFin": statCountNFin,
		prefixName+"countNaN":  statCountNaN,
		prefixName+"sum":       statSum,
		prefixName+"l1":        statL1,
		prefixName+"l2":        statL2,
		prefixName+"min":       statMin,
		prefixName+"mean":      statMean,
		prefixName+"median":    statMedian,
		prefixName+"max":       statMax,
		prefixName+"std":       statStd,
		prefixName+"var":       statVar,
		prefixName+"histo":     statHisto,
	}
