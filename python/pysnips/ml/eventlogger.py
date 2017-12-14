# -*- coding: utf-8 -*-

#
# Imports
#

import                        contextlib
import                        os
import                        numpy        as np
import                        re
import                        sys
import                        threading
import                        time
import                        uuid
try:
	from StringIO          import StringIO     as BytesIO
except ImportError:
	from io                import BytesIO

from   .tfevents       import *
from   .tfevents       import convert_metadata

__all__ = ["EventLogger", "NullEventLogger", "tagscope", "getEventLogger",
           "logScalar", "logScalars", "logImage",   "logAudio",   "logText",
           "logTensor", "logHist",    "logMessage", "logSession", "logLayout",
           "get_event_logger", "log_scalar",  "log_scalars", "log_image",
           "log_audio",        "log_text",    "log_tensor",  "log_hist",
           "log_message",      "log_session", "log_layout",
           "TfLogLevel", "TfSessionStatus", "TfDataType", "TfColorSpace"]


#
# Event Logger.
#

class EventLogger(object):
	"""
	Event logging to Protocol Buffers.
	"""
	
	def __init__           (self, logDir, step=None, flushSecs=60.0, flushBufSz=None, **kwargs):
		# Training logistics-related
		self._logDir       = str(logDir)
		self._creationStep = step
		self._currentStep  = 0 if step is None else int(self._creationStep)
		self._creationTime = time.time()
		self._summaryTime  = self._creationTime
		self._uuid         = uuid.uuid4()
		self._metadata     = set()
		self._values       = {}
		self.__bytebuffer = bytearray()
		assert     os.path.isdir (self._logDir)
		assert not os.path.isfile(self._logFilePath)
		
		# Threading-related
		self._flushSecs    = flushSecs
		self._flushBufSz   = flushBufSz
		self._flushThread  = None
		self._cond         = threading.Condition()
		self._tls          = threading.local()
		
		#
		# Append the "file header" of sorts, consisting of the fileVersion event
		# and possibly a TfSessionLog event.
		#
		# We avoid appendBuffer() because it will either flush or trigger the
		# spawning of the flusher. There are two good reasons not to do it now:
		#
		# 1) We will possibly append two records (fileVersion and session log)
		#    without the protection of the lock.
		# 2) It may potentially trigger the first write and thus creation
		#    of the log file; But if this logger doesn't end up getting used,
		#    or doesn't have a chance to be used before a crash, there's no
		#    point polluting the filesystem with embryonic, empty
		#    tfevents files and slowing TB down.
		#
		self.__bytebuffer += TfEvent(step        = self._currentStep,
		                             wallTime    = self._creationTime,
		                             fileVersion = "brain.Event:2").asRecordByteArray()
		if self._creationStep is not None:
			self.__bytebuffer += TfSessionLog(TfSessionStatus.START,      \
			                                  "Restarting...",            \
			                                  self._logDir)               \
			                     .asEvent(step     = self._currentStep,   \
			                              wallTime = self._creationTime)  \
			                     .asRecordByteArray()
	
	def __del__            (self):
		self.close()
	
	@property
	def _logFileName       (self):
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
		    self._creationTime,
		    self._uuid
		)
	
	@property
	def _logFilePath       (self):
		return os.path.join(self._logDir, self._logFileName)
	
	@property
	def _tagScopePath      (self):
		if not hasattr(self.__tls, "_tagScopePath"):
			self.__tls._tagScopePath = []
		return self.__tls._tagScopePath
	
	@property
	def asynchronous       (self):
		return (self._flushSecs is not None) and (self._flushSecs > 0.0)
	@property
	def overflowing        (self):
		return (self._flushBufSz is not None) and (len(self.__bytebuffer) > self._flushBufSz)
	@contextlib.contextmanager
	def tagscope           (self, *groupNames):
		"""
		Enter a named tag scope in the current thread.
		
		The tag subgroup names may not contain the "/" hierarchical separator.
		"""
		
		for groupName in groupNames:
			if(sys.version_info[0] <  3 and isinstance(groupName, unicode) or
			   sys.version_info[0] >= 3 and isinstance(groupName, str)):
				groupName = groupName.encode("utf-8")
			assert isinstance(groupName, str)
			assert groupName != ""
			assert "/" not in groupName
		for groupName in groupNames:
			self._tagScopePath.append(groupName)
		yield
		for groupName in groupNames:
			self._tagScopePath.pop()
	
	def getFullTag         (self, tag):
		"""
		Compute the fully-qualified tag given the partial tag provided, taking
		into account the tag scopes defined so far in the current thread.
		"""
		
		assert not tag.startswith("/")
		tag = re.sub("/+", "/", tag)
		tag = "/".join(self._tagScopePath+[tag])
		return tag
	
	@classmethod
	def getEventLogger     (kls):
		"""
		Return the default logger for the current thread.
		
		This will return the logger currently top-of-stack.
		"""
		
		l = EventLogger.__tls._loggerStack = EventLogger.__tls.__dict__.get("_loggerStack", [])
		return l[-1] if l else NullEventLogger()
	
	def __enter__          (self):
		"""
		Make this event logger the default logger for the current thread.
		
		This is done by pushing it onto the stack of loggers.
		"""
		
		l = EventLogger.__tls._loggerStack = EventLogger.__tls.__dict__.get("_loggerStack", [])
		l.append(self)
		return self
	def __exit__           (self, *exc):
		"""
		Remove this event logger as the default logger for the current thread.
		
		This is done by popping it from the stack of loggers on context exit.
		"""
		
		self.close()
		assert EventLogger.__tls._loggerStack.pop() == self
	
	def _spawnFlushThread  (self):
		"""
		Spawn the flusher thread, if it hasn't been spawned already, and if we
		have asked for asynchronous writing.
		"""
		
		with self._cond:
			if self.asynchronous and not self._flushThread:
				def flusher():
					"""
					Flusher thread implementation. Simply waits on the
					condition and periodically flush to disk the buffer
					contents.
					
					*MUST NOT* call close(), because it *WILL* lead to deadlock.
					
					On exit request, will indicate its own exit by destroying
					the reference to itself in the EventLogger, and will then
					notify all waiters before terminating.
					"""
					
					thrd = threading.currentThread()
					with self._cond:
						while not thrd.isExiting:
							self._cond.wait(self._flushSecs)
							self.flush()
						self._flushThread = None
						self._cond.notifyAll()
				
				
				thrdName = "eventlogger-{:x}-flushThread".format(id(self))
				self._flushThread = threading.Thread(target=flusher, name=thrdName)
				self._flushThread.isExiting = False
				self._flushThread.start()
		
		return self
	
	def close              (self):
		"""
		Close the summary writer.
		
		Specifically, causes the flusher thread to terminate, then wait for a
		clean exit.
		"""
		
		with self._cond:
			while self._flushThread:
				self._flushThread.isExiting = True
				self._cond.notifyAll()
				self._cond.wait()
		return self.flush()
	
	def step               (self, step=None):
		"""
		Increment (or set) the global step number.
		"""
		
		with self._cond:
			if step is None:
				"""
				Since the step number is being changed, enqueue all of the
				waiting summaries to the bytebuffer, so that they are
				recorded with the correct step number.
				"""
				self.appendSummary()
				self._currentStep += 1
			else:
				"""
				We're forcibly changing the global step number. We should
				flush out the current buffers. This includes the enqueueing
				of summaries.
				"""
				self.flush()
				self._currentStep = int(step)
		return self
	
	def appendBuffer       (self, b):
		"""
		Append to the bytebuffer.
		
		**All** additions to the bytebuffer **must** happen through this
		method. This method and the bytebuffer are the portal between the
		log-generating and log-writing parts of the object.
		
		The only time the bytebuffer's size can change other than through this
		method is when it is flushed and emptied periodically from within the
		flush() method, which may be called either synchronously by any thread,
		or asynchronously by the flusher thread.
		"""
		
		with self._cond:
			self.__bytebuffer += bytearray(b)
			if self.overflowing: self.flush()
			else:                self._spawnFlushThread()
		return self
	
	def appendEvent        (self, e):
		"""
		Append a TfEvent to the bytebuffer.
		"""
		
		with self._cond:
			self.appendBuffer(e.asRecordByteArray())
		return self
	
	def appendSummary      (self):
		"""
		Withdraw every value from the dictionary, construct a summary message
		from them and enqueue it for writeout, but do not flush the bytebuffer.
		"""
		
		with self._cond:
			if self._values:
				self.appendEvent(TfSummary(self._values)
				                 .asEvent(step     = self._currentStep,
				                          wallTime = self._summaryTime))
				self._values = {}
		return self
	
	def __maybeForceAppend (self, tag):
		if tag in self._values:
			self.appendSummary()
		return self
	
	def recordValue        (self, tag, val):
		"""
		Records a value in the values summary.
		"""
		
		assert isinstance(tag, str)
		assert isinstance(val, TfValue)
		with self._cond:
			if hasattr(val, "metadata") and tag in self._metadata:
				del val.metadata
			self._values[tag] = val
			self._metadata.add(tag)
			self._summaryTime = time.time()
		return self
	
	def flush              (self):
		"""
		Write out and flush the bytebuffer to disk synchronously.
		"""
		
		with self._cond:
			self.appendSummary()
			if self.__bytebuffer:
				with open(self._logFilePath, "ab") as f:
					f.write(self.__bytebuffer)
					f.flush()
				self.__bytebuffer = bytearray()
		return self
	
	def logScalar          (self, tag, scalar,
	                        displayName=None,
	                        description=None,
	                        pluginName=None,
	                        pluginContent=None):
		"""Log a single scalar value."""
		tag      = self.getFullTag(tag)
		metadata = convert_metadata(displayName, description, pluginName, pluginContent)
		val      = TfValue(tag, simpleValue=float(scalar), metadata=metadata)
		
		with self._cond:
			self.__maybeForceAppend(tag).recordValue(tag, val)
		return self
	
	def logScalars         (self, scalarsDict):
		with self._cond:
			for tag, scalar in scalarsDict:
				self.logScalar(tag, scalar)
		return self
	
	def logImage           (self, tag, image, csc=None, h=None, w=None,
	                        displayName=None,
	                        description=None,
	                        pluginName=None,
	                        pluginContent=None,
	                        maxOutputs=3):
		"""
		Log image(s).
		
		Accepts either a single encoded image as a bytes/bytearray, or one or
		more images as a 3- or 4-D numpy array in the form CHW or NCHW.
		"""
		tag      = self.getFullTag(tag)
		metadata = convert_metadata(displayName, description, pluginName, pluginContent)
		if   isinstance(image, (bytes, bytearray)):
			#
			# "Raw" calling convention: `image` contains an image file, and all
			# arguments are mandatory.
			#
			csc, w, h = int(csc), int(w), int(h)
		elif isinstance(image, np.ndarray):
			#
			# "Numpy" calling convention: `image` is a numpy ndarray shaped (C,H,W).
			# Conversion is to PNG -z 9. The precise transformation depends on the
			# number of channels and the datatype:
			#
			# If   c == 1: Assume grayscale.
			# Elif c == 2: Assume grayscale+alpha.
			# Elif c == 3: Assume RGB.
			# Elif c == 4: Assume RGBA.
			# Else: raise
			#
			# If   dtype == np.uint8:  Assume 8-bit unsigned [0, 255].
			# If   dtype == np.float*: Assume floating-point [0,   1].
			# Else: raise
			#
			
			c, h, w = image.shape
			if image.dtype != np.uint8:
				image = (image*255.0).astype("uint8")
			
			if   c == 1:
				csc  = TfColorSpace.GRAYSCALE
				mode = "L"
			elif c == 2:
				csc = TfColorSpace.GRAYSCALE_ALPHA
				mode = "LA"
			elif c == 3:
				csc = TfColorSpace.RGB
				mode = "RGB"
			elif c == 4:
				csc = TfColorSpace.RGBA
				mode = "RGBA"
			else:
				raise ValueError("Invalid image specification!")
			
			#
			# Encode as PNG using an in-memory buffer as the "file" stream.
			#
			
			from PIL.Image import frombytes
			stream = BytesIO()
			image  = frombytes(mode, (w,h), image.transpose(1,2,0).copy().data)
			image.save(stream, format="png", optimize=True)  # Always PNG -z 9
			image = stream.getvalue()
			stream.close()
		else:
			raise ValueError("Unable to interpret image arguments!")
		val = TfImage(h, w, csc, image).asValue(tag, metadata)
		
		with self._cond:
			self.__maybeForceAppend(tag).recordValue(tag, val)
		return self
	
	def logAudio           (self, tag, audio, sampleRate, numChannels=1,
	                        displayName=None,
	                        description=None,
	                        pluginName=None,
	                        pluginContent=None,
	                        maxOutputs=3):
		"""
		Log audio sample(s).
		
		Accepts either a single or a batch of audio samples as a numpy 1-D or
		2-D array of 16-bit signed integers, and encodes it to WAVE format.
		"""
		
		tag      = self.getFullTag(tag)
		metadata = convert_metadata(displayName, description, pluginName, pluginContent)
		lengthFrames = len(audio)
		
		#
		# Always encode the audio as 16-bit integer WAVE.
		#
		
		import wave
		stream = BytesIO()
		wavewr = wave.open(stream, "wb")
		wavewr.setnchannels(numChannels)
		wavewr.setframerate(sampleRate)
		wavewr.setsampwidth(2) # 16-bit integer
		wavewr.writeframes(audio)
		wavewr.close()
		audio = stream.getvalue()
		stream.close()
		val   = TfAudio(sampleRate,
		                numChannels,
		                lengthFrames,
		                audio,
		                "audio/wav").asValue(tag, metadata)
		
		with self._cond:
			self.__maybeForceAppend(tag).recordValue(tag, val)
		return self
	
	def logText            (self, tag, text,
	                        displayName=None,
	                        description=None,
	                        pluginName=None,
	                        pluginContent=None):
		tag      = self.getFullTag(tag)
		metadata = convert_metadata(displayName, description, pluginName, pluginContent)
		val      = TfTensor(text).asValue(tag, metadata)
		
		with self._cond:
			self.__maybeForceAppend(tag).recordValue(tag, val)
		return self
	
	def logTensor          (self, tag, tensor, dimNames=None,
	                        displayName=None,
	                        description=None,
	                        pluginName=None,
	                        pluginContent=None):
		tag      = self.getFullTag(tag)
		metadata = convert_metadata(displayName, description, pluginName, pluginContent)
		val      = TfTensor(tensor, dimNames).asValue(tag, metadata)
		
		with self._cond:
			self.__maybeForceAppend(tag).recordValue(tag, val)
		return self
	
	def logHist            (self, tag, tensor, bins=None,
	                        displayName=None,
	                        description=None,
	                        pluginName=None,
	                        pluginContent=None):
		tag      = self.getFullTag(tag)
		metadata = convert_metadata(displayName, description, pluginName, pluginContent)
		val      = TfHistogram(tensor, bins).asValue(tag, metadata)
		
		with self._cond:
			self.__maybeForceAppend(tag).recordValue(tag, val)
		return self
	
	def logMessage         (self, msg, level=TfLogLevel.UNKNOWN):
		with self._cond:
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
			
			self.appendSummary()
			self.appendEvent(TfLogMessage(msg, level)
			                 .asEvent(step=self._currentStep))
			self.flush()
		return self
	
	def logSession         (self, status, msg=None, path=None):
		with self._cond:
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
			
			self.appendSummary()
			self.appendEvent(TfSessionLog(status, msg, path)
			                 .asEvent(step=self._currentStep))
			self.flush()
		return self
	
	def logLayout          (self, layout):
		#
		# A layout is a TfTensor of TfDataType.STRING, where the string payload
		# is an encoded TfLayout protobuf message. The tensor is logged as a
		# TfValue with the magic tag "custom_scalars__config__" and the
		# pluginName "custom_scalars".
		#
		return self
	
	#
	# Static, thread-local data
	#
	# Stores the current thread's stack of default event loggers.
	#
	
	__tls       = threading.local()
	
	#
	# snake_case aliases for those who prefer that.
	#
	
	get_event_logger = getEventLogger
	log_scalar       = logScalar
	log_scalars      = logScalars
	log_image        = logImage
	log_audio        = logAudio
	log_text         = logText
	log_tensor       = logTensor
	log_hist         = logHist
	log_message      = logMessage
	log_session      = logSession
	log_layout       = logLayout


#
# Null Event Logger
#

class NullEventLogger(EventLogger):
	"""
	Null Event Logger
	
	Used when one needs a "null", "sink" or "/dev/null" EventLogger instance.
	
	It's identical in interface to an EventLogger, but it has been defanged.
	Null Event loggers will still support tag scopes and such, but will not log
	anything whatsoever to any file, nor spawn any logging thread to do so.
	"""
	
	def __init__           (self, *args, **kwargs):
		# Training logistics-related
		self._logDir       = "."
		self._currentStep  = 0
		self._creationStep = None
		self._creationTime = 0.0
		self._uuid         = uuid.UUID(int=0)
		self._metadata     = set()
		self._values       = {}
		self.__bytebuffer  = bytearray()
		
		# Threading-related
		self._flushSecs    = None
		self._flushBufSz   = None
		self._flushThread  = None
		self._cond         = threading.Condition()
		self._tls          = threading.local()
	def appendBuffer       (self, b):         return self
	def _spawnFlushThread  (self):            return self
	def recordValue        (self, tag, val):  return self
	def flush              (self):            return self
	def step               (self, step=None): return self
	def close              (self):            return self




#
# Global convenience functions exposing the logging API, using the default
# (top-of-stack) event logger.
#

@contextlib.contextmanager
def tagscope        (*args, **kwargs):
	with getEventLogger().tagscope(*args, **kwargs):
		yield
def getEventLogger  ():
	return EventLogger.getEventLogger()
def logScalar       (*args, **kwargs):
	return getEventLogger().logScalar (*args, **kwargs)
def logScalars      (*args, **kwargs):
	return getEventLogger().logScalars(*args, **kwargs)
def logImage        (*args, **kwargs):
	return getEventLogger().logImage  (*args, **kwargs)
def logAudio        (*args, **kwargs):
	return getEventLogger().logAudio  (*args, **kwargs)
def logText         (*args, **kwargs):
	return getEventLogger().logText   (*args, **kwargs)
def logTensor       (*args, **kwargs):
	return getEventLogger().logTensor (*args, **kwargs)
def logHist         (*args, **kwargs):
	return getEventLogger().logHist   (*args, **kwargs)
def logMessage      (*args, **kwargs):
	return getEventLogger().logMessage(*args, **kwargs)
def logSession      (*args, **kwargs):
	return getEventLogger().logSession(*args, **kwargs)
def logLayout       (*args, **kwargs):
	return getEventLogger().logLayout (*args, **kwargs)
def get_event_logger():
	return getEventLogger()
def log_scalar      (*args, **kwargs):
	return logScalar  (*args, **kwargs)
def log_scalars     (*args, **kwargs):
	return logScalars (*args, **kwargs)
def log_image       (*args, **kwargs):
	return logImage   (*args, **kwargs)
def log_audio       (*args, **kwargs):
	return logAudio   (*args, **kwargs)
def log_text        (*args, **kwargs):
	return logText    (*args, **kwargs)
def log_tensor      (*args, **kwargs):
	return logTensor  (*args, **kwargs)
def log_hist        (*args, **kwargs):
	return logHist    (*args, **kwargs)
def log_message     (*args, **kwargs):
	return logMessage (*args, **kwargs)
def log_session     (*args, **kwargs):
	return logSession (*args, **kwargs)
def log_layout      (*args, **kwargs):
	return logLayout  (*args, **kwargs)

