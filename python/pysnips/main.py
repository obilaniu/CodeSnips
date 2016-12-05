#!/usr/bin/env python
# -*- coding: utf-8 -*-

#
# Main Function.
#
# Invokes "verb functions" defined in the main module. Verb functions are
# simply functions called verb_<verbName>, and are invoked from the command-
# line as
#
#     python  script.py  {verbName} [otherArgs...]
#
# The following or similar should be put at the bottom of script.py:
#
# if __name__ == "__main__":
# 	import sys, main
# 	main.main(sys.argv)
#

def main(argv):
	#
	# This script is invoked using a verb that denotes the general action to
	# take, plus optional arguments. If no verb is provided or the one
	# provided does not match anything, invoke the "help" verb.
	#
	
	import __main__
	if     len(argv) <= 1:   return __main__.verb_help(argv)
	
	verbName   = argv[1]
	verbFnName = "verb_"+verbName
	verbFn     = getattr(__main__, verbFnName, None)
	
	if not callable(verbFn): return __main__.verb_help(argv)
	else:                    return verbFn(argv)
