#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os, re


class Experiment(object):
	"""
	Experiment.
	
	An experiment comprises both an in-memory state and an on-disk state. At
	regular intervals, the in-memory state is synchronized with the on-disk
	state, thus permitting a resume should the experiment be killed. These
	on-disk serializations are called "snapshots".
	
	The hierarchical organization of files within an experiment is as follows:
	
		Experiment
		*   datadir/                   | The folder where the datasets are found and loaded from.
		*   tempdir/                   | A folder on the local disk for temporary files.
		*   workdir/                   | The main working directory.
			->  snapshot/              | Snapshot directory
				-> <#>/                | Snapshot numbered <#>
					->  *.hdf5, ...    | Data files and suchlike.
			->  latest                 | Symbolic link to latest snapshot.
	
	The invariants to be preserved are:
		- workdir/latest either does not exist, or is a *symbolic link* pointing
		  to the directory `snapshot/#`, which does exist and has a complete
		  loadable snapshot within it.
		- Snapshots are not modified in any way after they've been dumped,
		  except for deletions due to purging.
	"""
	
	def __init__(self,
	             workDir=".",
	             dataDir=".",
	             tempDir=".",
	             *args, **kwargs):
		self.__workDir = os.path.abspath(workDir)
		self.__dataDir = os.path.abspath(dataDir)
		self.__tempDir = os.path.abspath(tempDir)
		self.__ready   = False
		
		self.mkdirp(self.workDir)
		self.mkdirp(self.snapDir)
		self.mkdirp(self.tempDir)
		
		self.__dict__.update(dict(filter(lambda k: not k[0].startswith("__"),
		                                 kwargs.iteritems())))
	
	
	# Fundamental properties
	@property
	def workDir(self): return self.__workDir
	@property
	def dataDir(self): return self.__dataDir
	@property
	def tempDir(self): return self.__tempDir
	@property
	def snapDir(self):
		return os.path.join(self.workDir, "snapshot")
	@property
	def latestLink(self):
		return os.path.join(self.workDir, "latest")
	@property
	def latestSnapshotNum(self):
		if self.haveSnapshots():
			s = os.readlink(self.latestLink)
			assert(os.path.dirname(s) == "snapshot")
			s = os.path.basename(s)
			s = int(s, base=10)
			assert(s >= 0)
			return s
		else:
			return -1
	@latestSnapshotNum.setter
	def latestSnapshotNum(self, n):
		assert isinstance(n, int)
		self.__markLatest(n)
	@property
	def nextSnapshotNum(self):
		return self.latestSnapshotNum+1
	@property
	def ready(self):
		return self.__ready
	
	
	# General utilities. Not to be overriden.
	def getWorkdirRelPathToSnapshot(self, n):
		if isinstance(n, str):
			n = int(n, base=10)
		return os.path.join("snapshot", str(n))
	def getFullPathToSnapshot      (self, n):
		return os.path.join(self.workDir, self.getWorkdirRelPathToSnapshot(n))
	
	
	# Private functions
	def __markReady(self):
		self.__ready = True
		return self
	
	def __markLatest(self, n):
		"""Atomically reroute the "latest" symlink in the working directory so
		that it points to the given snapshot number."""
		self.atomicSymlink(self.getWorkdirRelPathToSnapshot(n), self.latestLink)
		return self
	
	
	#
	# Mutable State Management.
	#
	# To be implemented by user as he/she sees fit.
	#
	
	def load(self, path):
		"""Load mutable state from given path.
		
		Returns `self`."""
		return self
	
	def dump(self, path):
		"""Dump mutable state to given path.
		
		The state can be saved as either
		- A file with exactly the name `path` or
		- A directory with exactly the name `path` containing a freeform
		  hierarchy underneath it.
		
		When invoked by the snapshot machinery, the path's basename as given
		by os.path.basename(path) will be the number this snapshot will be
		be assigned, and it is equal to self.nextSnapshotNum.
		
		Returns `self`."""
		return self
	
	def fromScratch(self):
		"""Start a fresh experiment, from scratch.
		
		Returns `self`."""
		
		assert(not os.path.lexists(self.latestLink) or
		           os.path.islink (self.latestLink))
		self.rmR(self.latestLink)
		return self.__markReady()
	
	def fromSnapshot(self, path):
		"""Start an experiment from a snapshot.
		
		Most likely, this method will invoke self.load(path) at an opportune
		time in its implementation.
		
		Returns `self`."""
		
		return self.__markReady()
	
	def run(self):
		"""Run a readied experiment.
		
		Returns `self`."""
		
		assert(self.ready)
		return self
	
	
	# High-level Snapshot & Rollback
	def snapshot(self):
		"""Take a snapshot of the experiment.
		
		Returns `self`."""
		nextSnapshotNum  = self.nextSnapshotNum
		nextSnapshotPath = self.getFullPathToSnapshot(nextSnapshotNum)
		
		if os.path.lexists(nextSnapshotPath):
			self.rmR(nextSnapshotPath)
		return self.dump(nextSnapshotPath).__markLatest(nextSnapshotNum)
	
	def rollback(self, n="latest"):
		"""Roll back the experiment to the given snapshot number.
		
		Returns `self`."""
		
		if n == "latest":
			if self.haveSnapshots(): return self.fromSnapshot(self.latestLink)
			else:                    return self.fromScratch()
		elif isinstance(n, int):
			loadSnapshotPath = self.getFullPathToSnapshot(n)
			assert(os.path.isdir(loadSnapshotPath))
			return self.__markLatest(n).fromSnapshot(loadSnapshotPath)
		else:
			raise ValueError("n must be int, or \"latest\"!")
	
	def haveSnapshots(self):
		"""Check if we have at least one snapshot."""
		return os.path.islink(self.latestLink) and os.path.isdir(self.latestLink)
	
	def purge(self, keep="latest"):
		"""Purge snapshot directory of all snapshots except a given list or set
		of them.
		
		Returns `self`."""
		
		assert(isinstance(keep, (list, set)) or keep=="latest")
		keep = [] if keep=="latest" else keep
		
		snaps, nonSnaps = self.listSnapshotDir(self.snapDir)
		snaps.difference_update(keep)
		
		snapsToDelete = snaps|nonSnaps
		for s in snapsToDelete:
			snapPath = os.path.join(self.snapDir, str(s))
			if(os.path.islink  (self.latestLink) and
			   os.path.samefile(self.latestLink, snapPath)):
				pass # Don't delete this snapshot, since it's the "latest"
			else:
				self.rmR(snapPath)
		
		return self
	
	
	# Filesystem Utilities
	@classmethod
	def mkdirp(kls, path):
		dirStack = []
		while not os.path.isdir(path):
			dirStack += [os.path.basename(path)]
			path      =  os.path.dirname (path)
		while dirStack:
			path = os.path.join(path, dirStack.pop())
			os.mkdir(path)
	
	@classmethod
	def isFilenameInteger(kls, name):
		return re.match("^(0|[123456789]\d*)$", name)
	
	@classmethod
	def listSnapshotDir(kls, path):
		entryList = os.listdir(path)
		
		snapshotSet    = set()
		nonsnapshotSet = set()
		for e in entryList:
			if kls.isFilenameInteger(e):
				snapshotSet.add(int(e, base=10))
			else:
				nonsnapshotSet.add(e)
		return snapshotSet, nonsnapshotSet
	
	@classmethod
	def rmR(kls, path):
		"""`rm -R path`. Deletes, but does not recurse into, symlinks.
		If the path does not exist, silently return."""
		if   os.path.islink(path) or os.path.isfile(path):
			os.unlink(path)
		elif os.path.isdir(path):
			walker = os.walk(path, topdown=False, followlinks=False)
			for dirpath, dirnames, filenames in walker:
				for f in filenames:
					os.unlink(os.path.join(dirpath, f))
				for d in dirnames:
					os.rmdir (os.path.join(dirpath, d))
			os.rmdir(path)
	
	@classmethod
	def atomicSymlink(kls, target, name):
		"""Same syntax as os.symlink, except that the new link called `name`
		will first be created with the `name` and `target`
		
		    `name.ATOMIC` -> `target`
		
		, then be atomically renamed to
		
		    `name` -> `target`
		
		, thus overwriting any previous symlink there. If a filesystem entity
		called `name.ATOMIC` already exists, it will be forcibly removed.
		"""
		
		linkAtomicName = name+".ATOMIC"
		linkFinalName  = name
		linkTarget     = target
		
		if os.path.lexists(linkAtomicName):
			kls.rmR(linkAtomicName)
		os.symlink(linkTarget,     linkAtomicName)
		
		################################################
		######## FILESYSTEM LINEARIZATION POINT ########
		######## vvvvvvvvvvvvvvvvvvvvvvvvvvvvvv ########
		os.rename (linkAtomicName, linkFinalName)
		######## ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ ########
		######## FILESYSTEM LINEARIZATION POINT ########
		################################################

