#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Imports.
import sys; sys.path += ["."] # Or whatever.
import argparse                             as Ap
import logging                              as L
import numpy                                as np
import os, pdb, sys
from   pysnips.ml.argparseactions import OptimizerAction
import time

__version__ = "0.0.0"



#
# Message Formatter
#

class MsgFormatter(L.Formatter):
	"""Message Formatter
	
	Formats messages with time format YYYY-MM-DD HH:MM:SS.mmm TZ
	"""
	
	def formatTime(self, record, datefmt):
		t           = record.created
		timeFrac    = abs(t-long(t))
		timeStruct  = time.localtime(record.created)
		timeString  = ""
		timeString += time.strftime("%F %T", timeStruct)
		timeString += "{:.3f} ".format(timeFrac)[1:]
		timeString += time.strftime("%Z",    timeStruct)
		return timeString



#############################################################################################################
##############################                   Subcommands               ##################################
#############################################################################################################

class Subcommand(object):
	name  = None
	
	@classmethod
	def addArgParser(cls, subp, *args, **kwargs):
		argp = subp.add_parser(cls.name, usage=cls.__doc__, *args, **kwargs)
		cls.addArgs(argp)
		argp.set_defaults(__subcmdfn__=cls.run)
		return argp
	
	@classmethod
	def addArgs(cls, argp):
		pass
	
	@classmethod
	def run(cls, d):
		pass


class Screw(Subcommand):
	"""Screw around with me in Screw(Subcommand)."""
	name = "screw"
	
	@classmethod
	def run(cls, d):
		print(cls.__doc__)


class Train(Subcommand):
	name = "train"
	
	LOGLEVELS = {"none":L.NOTSET, "debug": L.DEBUG, "info": L.INFO,
	             "warn":L.WARN,   "err":   L.ERROR, "crit": L.CRITICAL}
	
	
	@classmethod
	def addArgs(cls, argp):
		argp.add_argument("-w", "--workDir",        default=".",                type=str,
		    help="Path to the working directory for this experiment.")
		argp.add_argument("-d", "--dataDir",        default=".",                type=str,
		    help="Path to datasets directory.")
		argp.add_argument("-t", "--tempDir",        default=".",                type=str,
		    help="Path to temporary directory.")
		argp.add_argument("-l", "--loglevel",       default="info",             type=str,
		    choices=cls.LOGLEVELS.keys(),
		    help="Logging severity level.")
		argp.add_argument("-s", "--seed",           default=0x6a09e667f3bcc908, type=long,
		    help="Seed for PRNGs. Default is 64-bit fractional expansion of sqrt(2).")
		argp.add_argument("--summary",     action="store_true",
		    help="""Print a summary of the network.""")
		argp.add_argument("--model",                default="complex",          type=str,
		    choices=["real", "complex"],
		    help="Model Selection.")
		argp.add_argument("--dataset",              default="cifar10",          type=str,
		    choices=["cifar10", "cifar100", "svhn"],
		    help="Dataset Selection.")
		argp.add_argument("--dropout",              default=0,                  type=float,
		    help="Dropout probability.")
		argp.add_argument("-n", "--num-epochs",     default=200,                type=int,
		    help="Number of epochs")
		argp.add_argument("-b", "--batch-size",     default=64,                 type=int,
		    help="Batch Size")
		argp.add_argument("--act",                  default="relu",             type=str,
		    choices=["relu"],
		    help="Activation.")
		argp.add_argument("--aact",                 default="modrelu",          type=str,
		    choices=["modrelu"],
		    help="Advanced Activation.")
		argp.add_argument("--cuda",                 default=None,               type=int,
		    nargs="?", const=0,
		    help="CUDA device to use.")
		optp = argp.add_argument_group("Optimizers", "Tunables for all optimizers")
		optp.add_argument("--optimizer", "--opt", action=OptimizerAction,
		    type=str, default="nag", help="Optimizer selection.")
		optp.add_argument("--clipnorm", "--cn",     default=1.0,                type=float,
		    help="The norm of the gradient will be clipped at this magnitude.")
		optp.add_argument("--clipval",  "--cv",     default=1.0,                type=float,
		    help="The values of the gradients will be individually clipped at this magnitude.")
		optp.add_argument("--l1",                   default=0,                  type=float,
		    help="L1 penalty.")
		optp.add_argument("--l2",                   default=0,                  type=float,
		    help="L2 penalty.")
		optp.add_argument("--decay",                default=0,                  type=float,
		    help="Learning rate decay for optimizers.")
	
	@classmethod
	def run(cls, d):
		if not os.path.isdir(d.workDir):
			os.mkdir(d.workDir)
		
		logDir = os.path.join(d.workDir, "logs")
		if not os.path.isdir(logDir):
			os.mkdir(logDir)
		
		logFormatter      =   MsgFormatter ("[%(asctime)s ~~ %(levelname)-8s] %(message)s")
		
		stdoutLogSHandler = L.StreamHandler(sys.stdout)
		stdoutLogSHandler   .setLevel      (cls.LOGLEVELS[d.loglevel])
		stdoutLogSHandler   .setFormatter  (logFormatter)
		defltLogger       = L.getLogger    ()
		defltLogger          .setLevel     (cls.LOGLEVELS[d.loglevel])
		defltLogger          .addHandler   (stdoutLogSHandler)
		
		trainLogFilename  = os.path.join(d.workDir, "logs", "train.txt")
		trainLogFHandler  = L.FileHandler  (trainLogFilename, "a", "UTF-8", delay=True)
		trainLogFHandler     .setLevel     (cls.LOGLEVELS[d.loglevel])
		trainLogFHandler     .setFormatter (logFormatter)
		trainLogger       = L.getLogger    ("train")
		trainLogger          .setLevel     (cls.LOGLEVELS[d.loglevel])
		trainLogger          .addHandler   (trainLogFHandler)
		
		entryLogFilename  = os.path.join(d.workDir, "logs", "entry.txt")
		entryLogFHandler  = L.FileHandler  (entryLogFilename, "a", "UTF-8", delay=True)
		entryLogFHandler     .setLevel     (cls.LOGLEVELS[d.loglevel])
		entryLogFHandler     .setFormatter (logFormatter)
		entryLogger       = L.getLogger    ("entry")
		entryLogger          .setLevel     (cls.LOGLEVELS[d.loglevel])
		entryLogger          .addHandler   (entryLogFHandler)
		
		np.random.seed(d.seed % 2**32)
		
		import expmt;
		expmt.getExperiment(d).rollback().run()




#############################################################################################################
##############################               Argument Parsers               #################################
#############################################################################################################

def getArgParser(prog):
	argp = Ap.ArgumentParser(prog        = prog,
	                         usage       = None,
	                         description = None,
	                         epilog      = None,
	                         version     = __version__)
	subp = argp.add_subparsers()
	argp.set_defaults(argp=argp)
	argp.set_defaults(subp=subp)
	
	# Add global args to argp here?
	# ...
	
	
	# Add subcommands
	for v in globals().itervalues():
		if(isinstance(v, type)       and
		   issubclass(v, Subcommand) and
		   v != Subcommand):
			v.addArgParser(subp)
	
	# Return argument parser.
	return argp



#############################################################################################################
##############################                      Main                   ##################################
#############################################################################################################

def main(argv):
	sys.setrecursionlimit(10000)
	d = getArgParser(argv[0]).parse_args(argv[1:])
	return d.__subcmdfn__(d)
if __name__ == "__main__":
	main(sys.argv)

