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
           "TfLogLevel", "TfSessionStatus", "TfDataType", "TfColorSpace",
           "TfGraphKeys"]


#
# Event Logger.
#

class EventLogger(object):
	"""
	Event logging to Protocol Buffers.
	"""
	
	def __init__            (self, logDir,
	                               globalStep        = None,
	                               flushSecs         = 5.0,
	                               flushBufSz        = None,
	                               tagMatcher        = None,
	                               collectionMatcher = "^"+TfGraphKeys.SUMMARIES+"$"):
		"""
		EventLogger
		
		globalStep: Initial value of the globalStep to begin at.
		"""
		
		# Training logistics-related
		self._tagMatcher        = re.compile(tagMatcher)       .match if isinstance(tagMatcher,        str) else tagMatcher
		self._collectionMatcher = re.compile(collectionMatcher).match if isinstance(collectionMatcher, str) else collectionMatcher
		self._logDir            = str(logDir)
		self._startStep         = globalStep
		self._globalStep        = 0 if globalStep is None else int(self._startStep)
		self._logFileTime       = time.time()
		self._uuid              = uuid.uuid4()
		self._MB                = set()              # Metadata Buffer
		self._VB                = dict()             # Value    Buffer
		self._BB                = bytearray()        # Byte     Buffer
		
		# Threading-related
		self._flushSecs         = flushSecs
		self._flushBufSz        = flushBufSz
		self._flushThread       = None
		self._lock              = threading.Condition()
		self._tls               = threading.local()
		
		#
		# With the object mostly initialized and possessing all of the fields
		# it should, assert the existence of the log directory and nonexistence
		# of the file we're about to create.
		#
		self._initAsserts()
		
		#
		# Append the "file header" of sorts, consisting of the fileVersion event
		# and possibly a TfSessionLog event. We do *NOT* flush.
		#
		# We avoid write() because it will either flush or trigger the
		# spawning of the flusher. There is a good reason not to flush now:
		# Triggering the first write and thus the creation of the log file may
		# be undesirable if this logger doesn't end up getting used, or doesn't
		# have a chance to be used before a crash. In that case we would be
		# polluting the filesystem with embryonic, empty tfevents files and
		# slowing TensorBoard down.
		#
		eStep = self.globalStep
		eTime = self.logFileTime
		self.writeUnf(TfEvent(eStep, eTime, fileVersion="brain.Event:2")
		              .asRecordByteArray())
		if self.startStep is not None:
			self.writeUnf(TfSessionLog(TfSessionStatus.START,
			                           "Restarting...",
			                           self.logDir)
			              .asEvent(eStep, eTime)
			              .asRecordByteArray())
	
	def __del__             (self):
		self.close()
	
	def __enter__           (self):
		"""
		Make this event logger the default logger for the current thread.
		
		This is done by pushing it onto the stack of loggers.
		"""
		
		return self.push()
	def __exit__            (self, *exc):
		"""
		Remove this event logger as the default logger for the current thread.
		
		This is done by popping it from the stack of loggers on context exit.
		"""
		
		popped = self.close().pop()
		assert popped == self
	
	
	#
	# Fundamental, readonly properties.
	#
	@property
	def logDir              (self):
		"""
		Return log directory.
		"""
		
		return self._logDir
	
	@property
	def startStep           (self):
		return self._startStep
	
	@property
	def globalStep          (self):
		"""
		Current global step number of logging.
		"""
		return self._globalStep
	
	@property
	def logFileName         (self):
		"""
		Filename this event logger will log to.
		
		A name made of a zero-padded, 30-digit, nanosecond-resolution POSIX
		time plus a UUID4 in that order is
		
		    1) Constant-length for the next 100 million years
		    2) Unique with extremely high probability even when one of the
		       entropy sources (time or RNG) is broken, but not when both are
		       (e.g. when RNG is based on time)
		"""
		
		return "tfevents.{:030.9f}.{:s}.out".format(self.logFileTime, self.uuid)
	
	@property
	def logFilePath         (self):
		return os.path.join(self.logDir, self.logFileName)
	
	@property
	def logFileTime         (self):
		"""
		Timestamp of the earliest events in the file.
		
		This timestamp is also incorporated into the filename.
		"""
		
		return self._logFileTime
	
	@property
	def uuid                (self):
		return self._uuid
	
	@property
	def asynchronous        (self):
		return (self._flushSecs  is not None) and (self._flushSecs        > 0.0)
	
	@property
	def overflowing         (self):
		return (self._flushBufSz is not None) and (len(self._BB) > self._flushBufSz)
	
	@property
	def _tagScopeStack      (self):
		"""
		Returns the *current thread's* nested tag scopes *only*.
		
		In particular, a debugger thread *will not* see the same values as the
		debuggee thread.
		"""
		
		return self._tls.__dict__.setdefault("tagScopeStack", [])
	
	
	#
	# Context managers.
	#
	@contextlib.contextmanager
	def tagscope            (self, *groupNames):
		"""
		Enter a named tag scope in the current thread.
		
		The tag subgroup names may not contain the "/" hierarchical separator.
		"""
		
		namesPushed = 0
		for groupName in groupNames:
			try:
				self.pushTag(groupName)
				namesPushed += 1
			except:
				# Unwind the names partially stacked, then repropagate exception
				exc_info = sys.exc_info()
				for groupName in groupNames[:namesPushed:-1]:
					self.popTag()
				raise exc_info
		
		yield
		
		for groupName in groupNames:
			self.popTag()
	
	
	#
	# Internals. Do not touch.
	#
	def _initAsserts        (self):
		"""
		Check that the log directory we want does exist but that the log file
		does not.
		
		Exists only so it can be overriden and disabled for NullEventLogger.
		"""
		assert     os.path.isdir (self.logDir)
		assert not os.path.isfile(self.logFilePath)
		
		return self
	
	def _commonTagLogic     (self, **kwargs):
		"""
		Handle several flags common to most summary log*() methods and
		return a tuple metadata, reject, tag.
		"""
		
		metadata, reject, tag = None, True, None
		
		"""
		Currently we handle the following several pieces of information here:
		"""
		tag           = kwargs.pop("tag")
		collections   = kwargs.pop("collections",   [TfGraphKeys.SUMMARIES])
		tagPrefix     = kwargs.pop("tagPrefix",     None)
		displayName   = kwargs.pop("displayName",   None)
		description   = kwargs.pop("description",   None)
		pluginName    = kwargs.pop("pluginName",    None)
		pluginContent = kwargs.pop("pluginContent", None)
		
		"""
		If the tag prefix is None or True, it is automatically computed from
		the tag scopes. Otherwise, the given tag scope is used.
		"""
		tag = self.getFullyQualifiedTag(tag, tagPrefix)
		
		"""
		Construct metadata object if needed.
		"""
		metadata = convert_metadata(displayName,
		                            description,
		                            pluginName,
		                            pluginContent)
		
		"""
		If this tag does not belong to a selected collection, recommend
		rejection of the summary.
		"""
		if self._collectionMatcher is not None:
			for collection in collections:
				if self._collectionMatcher(collection):
					break
			else:
				return metadata, reject, tag
		
		"""
		Check the full tag for a match. Recommend rejecting the summary if
		there is no match.
		"""
		if self._tagMatcher is not None and not self._tagMatcher(tag):
			return metadata, reject, tag
		
		"""
		Accept the tag.
		"""
		return metadata, False, tag    # reject=False
	
	def _spawnFlushThread   (self):
		"""
		Spawn the flusher thread, if it hasn't been spawned already, and if we
		have asked for asynchronous writing.
		"""
		
		with self._lock:
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
					with self._lock:
						while not thrd.isExiting:
							self._lock.wait(self._flushSecs)
							self.flush()
						self._flushThread = None
						self._lock.notifyAll()
				
				
				thrdName = "eventlogger-{:x}-flushThread".format(id(self))
				self._flushThread = threading.Thread(target=flusher, name=thrdName)
				self._flushThread.isExiting = False
				self._flushThread.start()
		
		return self
	
	
	#
	# Public API
	#
	def push                (self):
		"""
		Push ourselves onto the thread-local stack of loggers.
		"""
		
		l = EventLogger._tls.__dict__.setdefault("loggerStack", [])
		l.append(self)
		return self
	
	def pop                 (self):
		"""
		Pop ourselves off the thread-local stack of loggers.
		"""
		
		return EventLogger._tls.loggerStack.pop()
	
	def pushTag             (self, groupName):
		assert isinstance(groupName, str)
		assert groupName != ""
		assert "/" not in groupName
		self._tagScopeStack.append(groupName)
		return self
	
	def popTag              (self):
		self._tagScopeStack.pop()
		return self
	
	def flush               (self):
		"""
		Write out and flush the bytebuffer to disk synchronously.
		"""
		
		with self._lock:
			self.commitValues()
			if self._BB:
				with open(self.logFilePath, "ab") as f:
					f.write(self._BB)
					f.flush()
				self._BB = bytearray()
		return self
	
	def close               (self):
		"""
		Close the summary writer.
		
		Specifically, request the flusher thread to exit, then wait for it to
		do so cleanly, and flush anything left in the buffer synchronously.
		"""
		
		with self._lock:
			while self._flushThread:
				self._flushThread.isExiting = True
				self._lock.notifyAll()
				self._lock.wait()
		return self.flush()
	
	def step                (self, step=None):
		"""
		Increment (or set) the global step number.
		"""
		
		with self._lock:
			if step is None:
				"""
				Since the step number is being changed, enqueue all of the
				waiting summaries to the bytebuffer, so that they are
				recorded with the correct step number.
				"""
				self.commitValues()
				self._globalStep += 1
			else:
				"""
				We're forcibly changing the global step number. We should
				flush out the current buffers. This includes the commit of
				summary values.
				"""
				self.flush()
				self._globalStep = int(step)
		return self
	
	def writeUnf            (self, b):
		"""
		Write raw data to the bytebuffer without flushing.
		
		**All** additions to the bytebuffer **must** happen through this
		method. This method and the bytebuffer are the portal between the
		log-generating and log-writing parts of the logger object.
		
		The only time the bytebuffer's size can change other than through this
		method is when it is flushed and emptied periodically from within the
		flush() method, which may be called either synchronously by any thread,
		or asynchronously by the flusher thread.
		"""
		with self._lock:
			self._BB += bytearray(b)
		return self
	
	def write               (self, b):
		"""
		Write raw data to the bytebuffer.
		
		Identical to writeUnf(), but also flushes (or schedules to be flushed
		asynchronously) the byte buffer.
		"""
		
		with self._lock:
			self.writeUnf(b)
			if self.overflowing: self.flush()
			else:                self._spawnFlushThread()
		return self
	
	def writeEvent          (self, e):
		"""
		Append a TfEvent to the bytebuffer.
		"""
		
		with self._lock:
			self.write(e.asRecordByteArray())
		return self
	
	def writeSummary        (self, s):
		"""
		Write a TfSummary object to the bytebuffer.
		"""
		
		with self._lock:
			self.writeEvent(s.asEvent(self.globalStep))
		return self
	
	def commitValues        (self):
		"""
		Withdraw every value from the dictionary, construct a summary message
		from them and enqueue it for writeout, but do not flush the bytebuffer.
		"""
		
		with self._lock:
			if self._VB:
				self.writeSummary(TfSummary(self._VB))
				self._VB = {}
		return self
	
	def stageValue          (self, val):
		"""
		Stage a value in the values summary.
		"""
		
		assert isinstance(val, TfValue)
		
		with self._lock:
			if hasattr(val, "metadata") and val.tag in self._MB:
				#
				# TensorBoard only keeps the first metadata it sees to save space
				#
				del val.metadata
			if val.tag in self._VB:
				#
				# There is a value with the same tag already staged in the
				# summary value buffer. Commit it to the bytebuffer.
				#
				self.commitValues()
			self._VB[val.tag] = val
			self._MB.add(val.tag)
		return self
	
	def getFullyQualifiedTag(self, tag, tagPrefix=None):
		"""
		Compute the fully-qualified tag given the partial tag provided, taking
		into account the tag scopes defined so far in the current thread.
		"""
		
		assert not tag.startswith("/") and not tag.endswith("/")
		assert tagPrefix is None or isinstance(tagPrefix, str)
		
		if tagPrefix is None:
			tagPath = self._tagScopeStack + [tag]
		else:
			assert not tagPrefix.startswith("/")
			if tagPrefix == "":
				tagPath = [tag]
			else:
				tagPath = [tagPrefix, tag]
		
		return re.sub("/+", "/", "/".join(tagPath))
	
	@classmethod
	def getEventLogger      (kls):
		"""
		Return the default logger for the current thread.
		
		This will return the logger currently top-of-stack.
		"""
		
		l = EventLogger._tls.__dict__.setdefault("loggerStack", [])
		return l[-1] if l else NullEventLogger()
	
	
	#
	# Public API, Logging Methods.
	#
	def logScalar           (self, tag, scalar, **kwargs):
		"""Log a single scalar value."""
		
		metadata, reject, tag = self._commonTagLogic(tag=tag, **kwargs)
		if reject: return self
		
		val = TfValue(tag, simpleValue=float(scalar), metadata=metadata)
		return self.stageValue(val)
	
	def logScalars          (self, scalarsDict, **kwargs):
		"""Log multiple scalar values, provided as a (tag, value) iterable."""
		
		with self._lock:
			for tag, scalar in scalarsDict:
				self.logScalar(tag, scalar, **kwargs)
		return self
	
	def logImage            (self, tag, images, csc=None, h=None, w=None, maxOutputs=3, **kwargs):
		"""
		Log image(s).
		
		Accepts either a single encoded image as a bytes/bytearray, or one or
		more images as a 3- or 4-D numpy array in the form CHW or NCHW.
		"""
		
		if   isinstance(images, (bytes, bytearray)):
			"""
			"Raw" calling convention: `image` contains an image file, and all
			arguments are mandatory. Image is logged encoded as-is
			"""
			
			metadata, reject, tag = self._commonTagLogic(tag=tag+"/image", **kwargs)
			if reject: return self
			
			val = TfImage(int(h), int(w), int(csc), images).asValue(tag, metadata)
			return self.stageValue(val)
		elif isinstance(images, (list, np.ndarray)):
			"""
			"Numpy" calling convention: `image` is a numpy ndarray shaped (N,C,H,W).
			Conversion is to PNG -z 9. The precise transformation depends on the
			number of channels, datatype and content.
			"""
			
			#
			# Expand dimensionality
			#
			if isinstance(images, np.ndarray) and images.ndim == 3:
				images = images[np.newaxis, ...]
			
			#
			# Iterate.
			#
			for i, image in enumerate(images):
				#
				# Do not output more than the limit of images.
				#
				if i >= maxOutputs:
					break
				
				#
				# Follow TF naming algorithm for image batches.
				#
				if i == 0 and maxOutputs == 1:
					metadata, reject, tag = self._commonTagLogic(tag=tag+"/image",         **kwargs)
				else:
					metadata, reject, tag = self._commonTagLogic(tag=tag+"/image/"+str(i), **kwargs)
				if reject: continue
				
				#
				# Follow TF type-conversion algorithm for individual images.
				#
				# If   c == 1: Assume grayscale.
				# Elif c == 2: Assume grayscale+alpha.
				# Elif c == 3: Assume RGB.
				# Elif c == 4: Assume RGBA.
				# Else: raise
				#
				c, h, w = image.shape
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
				# (continued TF type-conversion algorithm for individual images)
				#
				# If   image.dtype == np.uint8:
				#     pass
				# Elif image.min() >= 0:
				#     image /= image.max()/255.0
				#     image  = image.astype(np.uint8)
				# Else:
				#     image.scale( s.t. min >= -127 and max <= 128 )
				#     image += 127
				#
				if   image.dtype == np.uint8:
					pass
				elif image.min() >= 0:
					image *= +255.0/image.max()
				else:
					fMin, fMax = abs(-127.0/image.min()), abs(+128.0/image.max())
					image *= np.minimum(fMin, fMax)
					image += +127.0
				image = image.astype(np.uint8)
				
				#
				# Encode as PNG using an in-memory buffer as the "file" stream.
				#
				
				from PIL.Image import frombytes
				stream = BytesIO()
				image  = frombytes(mode, (w,h), image.transpose(1,2,0).copy().data)
				image.save(stream, format="png", optimize=True)  # Always PNG -z 9
				image = stream.getvalue()
				stream.close()
				
				#
				# Log the image.
				#
				val = TfImage(int(h), int(w), int(csc), image).asValue(tag, metadata)
				self.stageValue(val)
		else:
			raise ValueError("Unable to interpret image arguments!")
		
		return self
	
	def logAudio            (self, tag, audios, sampleRate, maxOutputs=3, **kwargs):
		"""
		Log audio sample(s).
		
		Accepts either a single or a batch of audio samples as a Numpy 1-D, 2-D
		or 3-D array of floating-point numbers in the range [-1, +1], and
		encodes it to WAVE format.
		
		A 1-D array is assumed to be shaped (Time).
		A 2-D array is assumed to be shaped (Chann,Time).
		A 3-D array is assumed to be shaped (Batch,Chann,Time).
		"""
		
		#
		# Expand dimensionality
		#
		if   isinstance(audios, np.ndarray) and audios.ndim == 1:
			audios = audios[np.newaxis, np.newaxis, ...]
		elif isinstance(audios, np.ndarray) and audios.ndim == 2:
			audios = audios[np.newaxis,             ...]
		
		#
		# Iterate.
		#
		for i, audio in enumerate(audios):
			#
			# Do not output more than the limit of audios.
			#
			if i >= maxOutputs:
				break
			
			#
			# Follow TF naming algorithm for audio batches.
			#
			if i == 0 and maxOutputs == 1:
				metadata, reject, tag = self._commonTagLogic(tag=tag+"/audio",         **kwargs)
			else:
				metadata, reject, tag = self._commonTagLogic(tag=tag+"/audio/"+str(i), **kwargs)
			if reject: continue
			
			#
			# If audios is a list, we must ensure the presence of a channels axis.
			# Then, in WAV, audio frames are interleaved, so we must transpose to (T,C).
			# Lastly, we want to encode as 16-bit signed integer:
			#
			
			if audio.ndim == 1:
				audio = audio[np.newaxis, ...]
			audio  = audio.transpose()
			audio *= 32767.0
			audio  = audio.astype(np.int16)
			
			#
			# Always encode the audio as 16-bit integer WAVE.
			#
			import wave
			stream = BytesIO()
			wavewr = wave.open(stream, "wb")
			wavewr.setnchannels(audio.shape[0])
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
			
			#
			# Log the audio.
			#
			self.stageValue(val)
		
		return self
	
	def logText             (self, tag, text, **kwargs):
		"""
		Log a tensor of text strings to the "Text" dashboard.
		"""
		
		kwargs["pluginName"] = "text"
		metadata, reject, tag = self._commonTagLogic(tag=tag, **kwargs)
		if reject: return self
		
		val      = TfTensor(text).asValue(tag, metadata)
		return self.stageValue(val)
	
	def logTensor           (self, tag, tensor, dimNames=None, **kwargs):
		"""
		Log a tensor.
		"""
		
		metadata, reject, tag = self._commonTagLogic(tag=tag, **kwargs)
		if reject: return self
		
		val      = TfTensor(tensor, dimNames).asValue(tag, metadata)
		return self.stageValue(val)
	
	def logHist             (self, tag, tensor, bins=None, **kwargs):
		"""
		Log a histogram.
		"""
		
		metadata, reject, tag = self._commonTagLogic(tag=tag, **kwargs)
		if reject: return self
		
		val = TfHistogram(tensor, bins).asValue(tag, metadata)
		return self.stageValue(val)
	
	def logMessage          (self, msg, level=TfLogLevel.UNKNOWN):
		"""
		Log a message.
		"""
		
		with self._lock:
			#
			# As a special case, log messages always provoke the commit of
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
			
			self.commitValues()
			self.writeEvent(TfLogMessage(msg, level).asEvent(self.globalStep))
			self.flush()
		return self
	
	def logSession          (self, status, msg=None, path=None):
		"""
		Log a session status change.
		"""
		
		with self._lock:
			#
			# As a special case, session log messages always provoke the
			# commit of all accumulated summaries and their synchronous
			# flushing to disk immediately afterwards in order to ensure that:
			#
			#   1) All summaries recorded before a session status change are
			#      temporally ordered *before* it, consistent with the order
			#      they were generated in.
			#   2) Session log messages are sufficiently rare and important
			#      that they deserve immediate writeout.
			#
			
			self.commitValues()
			self.writeEvent(TfSessionLog(status, msg, path).asEvent(self.globalStep))
			self.flush()
		return self
	
	def logLayout           (self, layout):
		"""
		Log a custom scalars chart layout.
		
		layout must be a TfLayout object.
		"""
		
		#
		# The serialized form of a layout is a TfTensor of datatype
		# TfDataType.STRING, whose string payload is the encoded TfLayout
		# protobuf message. The tensor is logged as a TfValue with the magic
		# tag "custom_scalars__config__" and the pluginName "custom_scalars".
		#
		# FIXME: THIS IS BROKEN. Replace with an equivalent. Also, the code
		# below allows filtering out of the tag; We want this tensor summary to
		# be written out unconditionally.
		#
		return self.logTensor("custom_scalars__config__",
		                      layout.asByteArray(),
		                      pluginName="custom_scalars",
		                      tagPrefix="")
	
	#
	# Per-class, thread-local data
	#
	_tls             = threading.local()
	
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
	
	def __init__            (self, *args, **kwargs):
		if "logDir" in kwargs:  kwargs["logDir"] = ""
		elif len(args) > 0:     args             = ("",)+args[1:]
		else:                   args             = [""]
		super(NullEventLogger, self).__init__(*args, **kwargs)
	def _initAsserts        (self):    return self
	def _spawnFlushThread   (self):    return self
	def writeUnf            (self, b): return self
	def flush               (self):    return self
	def close               (self):    return self




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

