#!/usr/bin/env python
# -*- coding: utf-8 -*-

#
# Imports
#

import numpy                         as np
from   pysnips.ml.eventlogger    import *



#
# Mandelbrot generation
#

def mandelbrotLattice():
	x      = np.linspace(-2.4, +1.0, 1700)
	y      = np.linspace(+1.2, -1.2, 1200)
	r, c   = np.meshgrid(x, y)
	return   r*1.0 + c*1.0j

def mandelbrotStep(z, g):
	z = np.where(np.absolute(z) > 1e4, z, z*z+g)
	
	#
	# Logging can be done in nested scopes. The global methods log*() retrieve
	# a default logger from a stack (which is per-thread), and each logger
	# also manages a stack of tag scopes (per-logger, per-thread)
	#
	# Creating a tag scope causes all log*() functions and methods to prefix
	# the tag with the name of the scopes, separated with forward slashes (/).
	#
	# In the example below, two histogram metrics are collected with the tags
	#
	#     `/histo/grams/zMag` and
	#     `/histo/grams/zAng`
	#
	
	with tagscope("histo", "grams"):
		logHist("zMag", np.absolute(z),
		        displayName = "Z Magnitude",
		        description = "Histogram of complex number magnitudes.")
	
	with tagscope("histo"):
		with tagscope("grams"):
			logHist("zAng", np.angle(z),
			        bins        = np.linspace(-np.pi, +np.pi, 1000),
			        displayName = "Z Angle",
			        description = "Histogram of complex number angles.")
	
	return z

def mandelbrotEscTime(escTime, z, i):
	#
	# Boolean escape conditions.
	#
	# A point in a Mandelbrot set is deemed to have "escaped" if it reaches
	# a magnitude of >2, since it is then guaranteed to be ejected to
	# infinity. The escape time (in # of steps) can be used to plot a beautiful
	# visualization of the Mandelbrot set by making it correspond to colors
	# in a palette.
	#
	
	zMag             = np.absolute(z)
	currentlyEscaped = zMag    > 2
	alreadyEscaped   = escTime < i
	newlyEscaped     = currentlyEscaped & ~alreadyEscaped
	
	#
	# One can also put the hierarchy directly in the tag name.
	#
	logScalar("nested/scopes/escaped", np.mean(currentlyEscaped))
	
	
	# Smooth escape-time estimate
	antiNaN          = 4.0*~newlyEscaped
	smoothTime       = i+1 - np.log(0.5 * np.log(zMag+antiNaN)) / np.log(2)
	
	#
	# Masked computation of escape time:
	#
	# If   already escaped:
	#     pass
	# Elif newly escaped:
	#     set i+1 - log(0.5*log(|z|))/log(2)
	# Else
	#     set i+1
	#
	
	escTime = np.where(currentlyEscaped, escTime,    i+1)
	escTime = np.where(newlyEscaped,     smoothTime, escTime)
	
	# Vizualization
	d2      = (i+1)/2.0
	img     = escTime/(i+1)
	palette = escTime > d2 # If False, Black-to-Red; If True, Red-to-White.
	imgR    = np.where(palette, 1,                   escTime/d2)
	imgGB   = np.where(palette, (escTime-d2)/(d2+1), 0)
	imgR    = np.where(currentlyEscaped, imgR,  0)
	imgGB   = np.where(currentlyEscaped, imgGB, 0)
	img     = np.stack([imgR, imgGB, imgGB], axis=0)
	
	logImage("viz", img).close()
	
	##DEBUG: OpenCV viewing code
	#import cv2 as cv
	#cvImg = (img[::-1]*255).transpose(1,2,0).astype(np.uint8)
	#cv.imshow("Image", cvImg)
	#cv.waitKey(30)
	
	# Can put rank-0 tensors into the Scalars viewing pane with pluginName="scalars".
	logTensor("test/rank0tensor", img.astype("float32").sum(), pluginName="scalars")
	
	return escTime

#
# Run Mandelbrot for 50 steps.
#

def main():
	#
	# Initialize Mandelbrot iteration scheme
	#
	g       = mandelbrotLattice()
	z       = g.copy()
	escTime = np.zeros_like(z, dtype="float64")
	
	#
	# Run the Mandelbrot iterated function for 50 steps, recording the
	# "escape time" of every point.
	#
	for i in range(50):
		z       = mandelbrotStep   (z,          g)
		escTime = mandelbrotEscTime(escTime, z, i)
		
		#
		# The current top-of-stack (default) event logger can be retrieved with
		# the global function getEventLogger() or the class method
		# EventLogger.getEventLogger(), and it has methods that correspond to
		# the global log*() functions.
		#
		# Moreover, these methods support a fluent interface through
		# method call chaining:
		#
		#     logMessage("foo", foo).logScalar("bar", bar).logAudio("baz", baz)
		#
		
		getEventLogger().logMessage("Step {:2d} done!".format(i), TfLogLevel.INFO) \
		                .logMessage("Now we log a random scalar", TfLogLevel.WARN) \
		                .logScalar ("loss", 0.65 * 0.95**i)                        \
		                .step().flush()
		
		print("Step {:2d}".format(i))



if __name__ == "__main__":
	#
	# Once with logging. Will print to tfevents.TIMESTAMP.NANOSECONDS.G-U-I-D.out
	# every 60 seconds.
	#
	# If resuming from a snapshot, make SURE that `step` is set to the step #
	# at the last snapshot. ALL data with `step` greater than the one provided
	# here WILL BE IGNORED by TensorBoard ("orphaned").
	#
	
	with EventLogger(".", step=0, flushSecs=60.0):
		main()
	
	#
	# Once without logging (dummy). Will print nothing.
	#
	# Demonstrates that even in the absence of a current EventLogger, a dummy
	# one will be instantiated (NullEventLogger), which supports all of the
	# method calls of a real EventLogger, except that it will log nothing and
	# output nothing.
	#
	
	main()

