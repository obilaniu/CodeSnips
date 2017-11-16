# -*- coding: utf-8 -*-

#
# Imports
#

import                        PIL.Image
import                        contextlib
import                        os
import                        threading
import                        time
import                        uuid
try:
	from StringIO          import StringIO as BytesIO
except ImportError:
	from io                import BytesIO

from   .               import nanoemitter


class EventLogger(object):
	"""
	Event logging to Protocol Buffers.
	"""
	
	def __init__(self, path, step):
		self.__step         = int(step)
		self.__creationStep = self.__step
		self.__creationTime = time.time()
		self.__uuid         = uuid.uuid4()
		self.__tls          = threading.local()
		self.__lock         = threading.RLock()
		self.__summary      = {}
		self.__bytebuffer   = bytearray()
		
		self.filePath = os.path.join(path, self.fileName)
		
		assert     os.path.isdir(path)
		assert not os.path.isfile(self.filePath)
	
	@property
	def currentStep (self): return self.__step
	@property
	def uuid        (self): return str(self.__uuid)
	@property
	def fileName    (self):
		"""
		Filename this event logger will log to.
		
		A name made of a zero-padded, 30-digit, nanosecond-resolution POSIX
		time plus a UUID4 in that order is
		
		    1) Constant-length for the next 100 million years
		    2) Unique with extremely high probability even when one of the
		       entropy sources (time or RNG) is broken, but not when both are
		       (e.g. when RNG is based on time)
		"""
		
		return "tfevents.{:030.9f}.{:s}.out".format(
		    self.__creationTime,
		    self.uuid
		)
	@property
	def tagScopePath(self):
		if not hasattr(self.__tls, "tagScopePath"):
			self.__tls.tagScopePath = []
		return self.__tls.tagScopePath
	
	@contextlib.contextmanager
	def tagscope    (self, *groupNames):
		"""
		Enter a named tag scope.
		
		The tag subgroup names may not contain the "/" hierarchical separator.
		"""
		
		for groupName in groupNames:
			assert isinstance(str, groupName)
			assert groupName != ""
			assert "/" not in groupName
		for groupName in groupNames:
			self.tagScopePath.append(groupName)
		yield
		for groupName in groupNames:
			self.tagScopePath.pop()
	
	@contextlib.contextmanager
	def asDefault   (self):
		"""
		Make this event logger the default logger for the thread.
		
		This is done by pushing it onto the stack of loggers and popping it
		once the context exits.
		"""
		
		if not hasattr(EventLogger.__tls, "loggerStack"):
			EventLogger.__tls.loggerStack = []
		
		EventLogger.__tls.loggerStack.append(self)
		yield
		EventLogger.__tls.loggerStack.pop()
	
	def getFullTag  (self, tag):
		"""
		Compute the fully-qualified tag given the partial tag provided, taking
		into account the tag scopes defined so far.
		"""
		
		assert not tag.startswith("/")
		tag = re.sub("/+", "/", tag)
		return "/".join(self.tagScopePath+[tag])
	
	@classmethod
	def getDefault  (kls):
		"""
		Return the default logger for the current thread.
		
		This will return the logger currently top-of-stack.
		"""
		
		return kls.__tls.loggerStack[-1]
	def step(self, step=None):
		"""
		Increment (or set) the global step number.
		"""
		
		with self.__lock:
			if step is None:
				self.__step += 1
			else:
				#
				# We're forcibly changing the global step number. We should
				# flush out the current buffers.
				#
				self.flush()
				self.__step = int(step)
		return self
	
	def enqueueSummaries(self):
		"""
		Withdraw every summary from the dictionary, construct a summary message
		from them and enqueue it for writeout, but do not flush the bytebuffer.
		"""
		
		with self.__lock:
			if self.__summary:
				self.__bytebuffer += Summary(self.__summary)            \
				                     .asEvent(step = self.currentStep)  \
				                     .asRecordByteArray()
				self.__summary     = {}
		return self
	
	def flush(self):
		"""
		Write out and flush the bytebuffer to disk.
		"""
		
		with self.__lock, open(self.filePath, "ab") as f:
			self.enqueueSummaries()
			f.write(self.__bytebuffer)
			self.__bytebuffer = bytearray()
		
		return self
	
	def logScalar(self, tag, scalar,
	              displayName=None,
	              description=None,
	              pluginName=None,
	              pluginContent=None):
		tag = self.getFullTag(tag)
		with self.__lock:
			self.__summary[tag] = None
		return self
	
	def logImage(self, tag, image, csc,
	              displayName=None,
	              description=None,
	              pluginName=None,
	              pluginContent=None):
		"""Log a single RGB(A), uint8 numpy image."""
		tag = self.getFullTag(tag)
		
		size   = tuple(image.shape[:2])
		rgba   = image.shape[2] == 4
		image  = PIL.Image.frombytes("RGBA" if rgba else "RGB", size, image.copy().data)
		stream = BytesIO()
		image.save(stream, format="png", optimize=True)  # Always PNG -z 9
		stream.close()
		data   = stream.getvalue()
		
		with self.__lock:
			self.__summary[tag] = None
		return self
	
	def logAudio(self, tag, audio, sampleRate,
	              numChannels=1,
	              displayName=None,
	              description=None,
	              pluginName=None,
	              pluginContent=None):
		tag = self.getFullTag(tag)
		# Always "audio/wav"
		with self.__lock:
			self.__summary[tag] = None
		return self
	
	def logText(self, tag, text,
	              displayName=None,
	              description=None,
	              pluginName=None,
	              pluginContent=None):
		tag = self.getFullTag(tag)
		with self.__lock:
			self.__summary[tag] = None
		return self
	
	def logTensor(self, tag, tensor,
	              displayName=None,
	              description=None,
	              pluginName=None,
	              pluginContent=None):
		tag = self.getFullTag(tag)
		with self.__lock:
			self.__summary[tag] = None
		return self
	
	def logHist(self, tag, tensor,
	              bins=None,
	              displayName=None,
	              description=None,
	              pluginName=None,
	              pluginContent=None):
		tag = self.getFullTag(tag)
		with self.__lock:
			self.__summary[tag] = None
		return self
	
	def logMessage(self, msg, level=0):
		with self.__lock:
			#
			# As a special case, log messages always provoke the enqueuing of
			# all accumulated summaries and their synchronous flushing to disk
			# immediately afterwards in order to ensure that
			#
			#   1) Any summaries about which the message might be are temporally
			#      ordered *before* the log message, consistent with the order
			#      they were generated in.
			#   2) The log message and summaries are made immediately visible
			#      on-disk, to allow for debugging in case of a crash soon
			#      afterwards. Otherwise, the messages might be lost along with
			#      the rest of the in-memory bytebuffer.
			#
			
			self.enqueueSummaries()
			self.__bytebuffer += LogMessage(msg, level)            \
			                     .asEvent(step = self.currentStep) \
			                     .asRecordByteArray()
			self.flush()
		return self
	
	def logSession(self, status, msg=None, path=None):
		with self.__lock:
			#
			# As a special case, session log messages always provoke the
			# enqueuing of all accumulated summaries and their synchronous
			# flushing to disk immediately afterwards in order to ensure that:
			#
			#   1) All summaries recorded before a session status change are
			#      temporally ordered *before* it, consistent with the order
			#      they were generated in.
			#   2) Session log changes are sufficiently rare and important that
			#      they deserve immediate writeout.
			#
			
			self.enqueueSummaries()
			self.__bytebuffer += SessionLog(status, msg, path)     \
			                     .asEvent(step = self.currentStep) \
			                     .asRecordByteArray()
			self.flush()
		return self
	
	
	#
	# Static, thread-local data
	#
	# Stores the current thread's stack of default event loggers.
	#
	
	__tls = threading.local()



#
# Global convenience functions exposing the logging API, using the default
# (top-of-stack) event logger.
#

def tagscope  (*args, **kwargs):
	yield  EventLogger.getDefault().tagscope  (*args, **kwargs)
def logScalar (*args, **kwargs):
	return EventLogger.getDefault().logScalar (*args, **kwargs)
def logImage  (*args, **kwargs):
	return EventLogger.getDefault().logImage  (*args, **kwargs)
def logAudio  (*args, **kwargs):
	return EventLogger.getDefault().logAudio  (*args, **kwargs)
def logText   (*args, **kwargs):
	return EventLogger.getDefault().logText   (*args, **kwargs)
def logTensor (*args, **kwargs):
	return EventLogger.getDefault().logTensor (*args, **kwargs)
def logHist   (*args, **kwargs):
	return EventLogger.getDefault().logHist   (*args, **kwargs)
def logMessage(*args, **kwargs):
	return EventLogger.getDefault().logMessage(*args, **kwargs)
def logSession(*args, **kwargs):
	return EventLogger.getDefault().logSession(*args, **kwargs)


