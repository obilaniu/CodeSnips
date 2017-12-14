# -*- coding: utf-8 -*-

#
# Imports
#

import numpy                   as np
import struct
import sys
import time

from   pebble              import (PebbleMessage,
                                   tfcrc32c,
                                   enc_tagvalue,
                                   enc_float64,
                                   enc_fixed64,
                                   enc_fixed32)



__all__ = ["TfLogLevel", "TfSessionStatus", "TfDataType", "TfColorSpace",
           "TfEvent", "TfSummary", "TfLogMessage", "TfSessionLog",
           "TfTaggedRunMetadata", "TfValue", "TfSummaryMetadata",
           "TfPluginData", "TfImage", "TfHistogram", "TfAudio", "TfTensor",
           "TfTensorShape", "TfDim"]


#
# "Enums" with constants defined by TF
#

class TfLogLevel          (object):
	UNKNOWN      =  0
	DEBUGGING    = 10
	INFO         = 20
	WARN         = 30
	ERROR        = 40
	FATAL        = 50
	
	def __init__(self): raise

class TfSessionStatus     (object):
	UNSPECIFIED  =  0
	START        =  1
	STOP         =  2
	CHECKPOINT   =  3
	
	def __init__(self): raise

class TfDataType          (object):
	INVALID      =  0  # Not a legal value for DataType.  Used to indicate a DataType field has not been set.
	FLOAT        =  1  # Data types that all computation devices are expected to be
	DOUBLE       =  2  # capable to support.
	INT32        =  3
	UINT8        =  4
	INT16        =  5
	INT8         =  6
	STRING       =  7
	COMPLEX64    =  8  # Single-precision complex
	INT64        =  9
	BOOL         = 10
	QINT8        = 11  # Quantized int8
	QUINT8       = 12  # Quantized uint8
	QINT32       = 13  # Quantized int32
	BFLOAT16     = 14  # Float32 truncated to 16 bits.  Only for cast ops.
	QINT16       = 15  # Quantized int16
	QUINT16      = 16  # Quantized uint16
	UINT16       = 17
	COMPLEX128   = 18  # Double-precision complex
	HALF         = 19
	RESOURCE     = 20
	VARIANT      = 21  # Arbitrary C++ data types
	UINT32       = 22
	UINT64       = 23
	
	def __init__(self): raise

class TfColorSpace        (object):
	GRAYSCALE        = 1
	GRAYSCALE_ALPHA  = 2
	RGB              = 3
	RGBA             = 4
	DIGITAL_YUV      = 5
	BGRA             = 6
	
	def __init__(self): raise

#
# Message Hierarchy as defined by TF
#

class TfEvent             (PebbleMessage):
	def __init__(self, step=0, wallTime=None,
	             fileVersion=None,
	             graphDef=None,
	             summary=None,
	             logMessage=None,
	             sessionLog=None,
	             taggedRunMetadata=None,
	             metaGraphDef=None):
		self.step      = int(step)
		self.wall_time = time.time() if wallTime is None else float(wallTime)
		
		if   fileVersion       is not None: self.file_version        = str(fileVersion)
		elif graphDef          is not None: self.graph_def           = bytearray(graphDef)
		elif summary           is not None: self.summary             = summary
		elif logMessage        is not None: self.log_message         = logMessage
		elif sessionLog        is not None: self.session_log         = sessionLog
		elif taggedRunMetadata is not None: self.tagged_run_metadata = taggedRunMetadata
		elif metaGraphDef      is not None: self.meta_graph_def      = bytearray(metaGraphDef)
		else: raise ValueError("The event is empty!")
	
	def asByteArray(self):
		b = bytearray()
		if   hasattr(self, "wall_time"):            b += enc_tagvalue("double",  1, self.wall_time)
		if   hasattr(self, "step"):                 b += enc_tagvalue("int64",   2, self.step)
		
		if   hasattr(self, "file_version"):         b += enc_tagvalue("string",  3, self.file_version)
		elif hasattr(self, "graph_def"):            b += enc_tagvalue("bytes",   4, self.graph_def)
		elif hasattr(self, "summary"):              b += enc_tagvalue("message", 5, self.summary)
		elif hasattr(self, "log_message"):          b += enc_tagvalue("message", 6, self.log_message)
		elif hasattr(self, "session_log"):          b += enc_tagvalue("message", 7, self.session_log)
		elif hasattr(self, "tagged_run_metadata"):  b += enc_tagvalue("message", 8, self.tagged_run_metadata)
		elif hasattr(self, "meta_graph_def"):       b += enc_tagvalue("bytes",   9, self.meta_graph_def)
		else: raise ValueError("The event is empty!")
		
		return b
	
	def asRecordByteArray(self):
		payload = self.asByteArray()
		header  = enc_fixed64(len(payload))
		
		return header + enc_fixed32(tfcrc32c(header)) + payload + enc_fixed32(tfcrc32c(payload))

class TfSummary           (PebbleMessage):
	def __init__(self, values={}):
		self.value = [v for k,v in sorted(values.iteritems())]
	
	def asByteArray(self):
		b = bytearray()
		if   hasattr(self, "value"):
			for v in self.value:                    b += enc_tagvalue("message", 1, v)
		return b
	
	def asEvent(self, **kwargs):
		return TfEvent(summary=self, **kwargs)

class TfLogMessage        (PebbleMessage):
	def __init__(self, message, level=TfLogLevel.UNKNOWN):
		self.message = str(message)
		self.level   = int(level)
	
	def asByteArray(self):
		b = bytearray()
		if   hasattr(self, "level"):                b += enc_tagvalue("enum",    1, self.level)
		if   hasattr(self, "message"):              b += enc_tagvalue("string",  2, self.message)
		return b
	
	def asEvent(self, **kwargs):
		return TfEvent(logMessage=self, **kwargs)

class TfSessionLog        (PebbleMessage):
	def __init__(self, status, msg=None, path=None):
		self.status = status
		if msg  is not None: self.msg             = str(msg)
		if path is not None: self.checkpoint_path = str(path)
	
	def asByteArray(self):
		b = bytearray()
		if   hasattr(self, "status"):               b += enc_tagvalue("enum",    1, self.status)
		if   hasattr(self, "checkpoint_path"):      b += enc_tagvalue("string",  2, self.checkpoint_path)
		if   hasattr(self, "msg"):                  b += enc_tagvalue("string",  3, self.msg)
		return b
	
	def asEvent(self, **kwargs):
		return TfEvent(sessionLog=self, **kwargs)

class TfTaggedRunMetadata (PebbleMessage):
	def __init__(self, tag, runMetadata):
		self.tag          = str(tag)
		self.run_metadata = bytearray(runMetadata)
	
	def asByteArray(self):
		b = bytearray()
		if   hasattr(self, "tag"):                  b += enc_tagvalue("string",  1, self.tag)
		if   hasattr(self, "run_metadata"):         b += enc_tagvalue("bytes",   2, self.run_metadata)
		return b
	
	def asEvent(self, **kwargs):
		return TfEvent(taggedRunMetadata=self, **kwargs)

class TfValue             (PebbleMessage):
	def __init__(self, tag,
	             simpleValue=None,
	             image=None,
	             histo=None,
	             audio=None,
	             tensor=None,
	             metadata=None):
		self.tag = str(tag)
		if   metadata    is not None: self.metadata     = metadata
		if   simpleValue is not None: self.simple_value = float(simpleValue)
		elif image       is not None: self.image        = image
		elif histo       is not None: self.histo        = histo
		elif audio       is not None: self.audio        = audio
		elif tensor      is not None: self.tensor       = tensor
		else: raise ValueError("The value is empty!")
	
	def asByteArray(self):
		b = bytearray()
		if   hasattr(self, "tag"):                  b += enc_tagvalue("string",  1, self.tag)
		if   hasattr(self, "metadata"):             b += enc_tagvalue("message", 9, self.metadata)
		
		if   hasattr(self, "simple_value"):         b += enc_tagvalue("float",   2, self.simple_value)
		elif hasattr(self, "image"):                b += enc_tagvalue("message", 4, self.image)
		elif hasattr(self, "histo"):                b += enc_tagvalue("message", 5, self.histo)
		elif hasattr(self, "audio"):                b += enc_tagvalue("message", 6, self.audio)
		elif hasattr(self, "tensor"):               b += enc_tagvalue("message", 8, self.tensor)
		
		return b

class TfSummaryMetadata   (PebbleMessage):
	def __init__(self, displayName=None, summaryDescription=None, pluginData=None):
		if displayName        is not None: self.display_name        = str(displayName)
		if summaryDescription is not None: self.summary_description = str(summaryDescription)
		if pluginData         is not None: self.plugin_data         = pluginData
	
	def asByteArray(self):
		b = bytearray()
		if   hasattr(self, "plugin_data"):          b += enc_tagvalue("message", 1, self.plugin_data)
		if   hasattr(self, "display_name"):         b += enc_tagvalue("string",  2, self.display_name)
		if   hasattr(self, "summary_description"):  b += enc_tagvalue("string",  3, self.summary_description)
		return b

class TfPluginData        (PebbleMessage):
	def __init__(self, pluginName=None, content=None):
		if pluginName is not None: self.plugin_name = str(pluginName)
		if content    is not None: self.content     = str(content)
	
	def asByteArray(self):
		b = bytearray()
		if   hasattr(self, "plugin_name"):          b += enc_tagvalue("string",  1, self.plugin_name)
		if   hasattr(self, "content"):              b += enc_tagvalue("string",  2, self.content)
		return b

class TfImage             (PebbleMessage):
	def __init__(self, height, width, colorspace, imageData):
		self.height               = int(height)
		self.width                = int(width)
		self.colorspace           = int(colorspace)
		self.encoded_image_string = bytearray(imageData)
	
	def asByteArray(self):
		b = bytearray()
		if   hasattr(self, "height"):               b += enc_tagvalue("int32",   1, self.height)
		if   hasattr(self, "width"):                b += enc_tagvalue("int32",   2, self.width)
		if   hasattr(self, "colorspace"):           b += enc_tagvalue("int32",   3, self.colorspace)
		if   hasattr(self, "encoded_image_string"): b += enc_tagvalue("bytes",   4, self.encoded_image_string)
		return b
	
	def asValue(self, tag, metadata=None):
		return TfValue(tag, image=self, metadata=metadata)

class TfHistogram         (PebbleMessage):
	def __init__(self, tensor, bins=None):
		bins              = self.getDefaultBuckets() if bins is None else bins
		
		self.min_         = float(np.min(tensor))
		self.max_         = float(np.max(tensor))
		self.num          = float(np.prod(tensor.shape))
		self.sum_         = float(np.sum(tensor.astype("float64")))
		self.sum_squares  = float(np.sum(tensor.astype("float64")**2))
		self.bucket_limit = bins[1:]
		self.bucket       = np.histogram(tensor, bins)[0]
	
	@classmethod
	def getDefaultBuckets(kls):
		"""
		Compute the default histogram buckets used by TF.
		"""
		buckets = []
		lim     = 1e-12
		while lim<1e20:
			buckets += [lim]
			lim     *= 1.1
		return [np.finfo("float64").min] + [-l for l in buckets[::-1]] + [0] + buckets + [np.finfo("float64").max]
	
	def asByteArray(self):
		b = bytearray()
		if   hasattr(self, "min_"):                 b += enc_tagvalue("double",  1, self.min_)
		if   hasattr(self, "max_"):                 b += enc_tagvalue("double",  2, self.max_)
		if   hasattr(self, "num"):                  b += enc_tagvalue("double",  3, self.num)
		if   hasattr(self, "sum_"):                 b += enc_tagvalue("double",  4, self.sum_)
		if   hasattr(self, "sum_squares"):          b += enc_tagvalue("double",  5, self.sum_squares)
		if   hasattr(self, "bucket_limit"):
			c = bytearray()
			for d in self.bucket_limit:             c += enc_float64(float(d))
			b                                         += enc_tagvalue("packed",  6, c)
		if   hasattr(self, "bucket"):
			c = bytearray()
			for d in self.bucket:                   c += enc_float64(float(d))
			b                                         += enc_tagvalue("packed",  7, c)
		return b
	
	def asValue(self, tag, metadata=None):
		return TfValue(tag, histo=self, metadata=metadata)

class TfAudio             (PebbleMessage):
	def __init__(self, sampleRate, numChannels, lengthFrames, audioData, contentType):
		self.sample_rate          = sampleRate
		self.num_channels         = numChannels
		self.length_frames        = lengthFrames
		self.encoded_audio_string = audioData
		self.content_type         = contentType
	
	def asByteArray(self):
		b = bytearray()
		if   hasattr(self, "sample_rate"):          b += enc_tagvalue("float",   1, self.sample_rate)
		if   hasattr(self, "num_channels"):         b += enc_tagvalue("int64",   2, self.num_channels)
		if   hasattr(self, "length_frames"):        b += enc_tagvalue("int64",   3, self.length_frames)
		if   hasattr(self, "encoded_audio_string"): b += enc_tagvalue("bytes",   4, self.encoded_audio_string)
		if   hasattr(self, "content_type"):         b += enc_tagvalue("string",  5, self.content_type)
		return b
	
	def asValue(self, tag, metadata=None):
		return Value(tag, audio=self, metadata=metadata)

class TfTensor            (PebbleMessage):
	def __init__(self, tensor, dimNames=None):
		if    (sys.version_info[0] <  3 and isinstance(tensor, (str, unicode)) or
		       sys.version_info[0] >= 3 and isinstance(tensor, str)):
			#
			# A single string to encode as a tensor.
			#
			
			if(sys.version_info[0] <  3 and isinstance(tensor, unicode) or
			   sys.version_info[0] >= 3 and isinstance(tensor, str)):
				tensor = tensor.encode("utf-8")
			
			self.dtype          = TfDataType.STRING
			self.tensor_shape   = convert_shape_np2tf((1,), dimNames)
			self.version_number = 0
			self.tensor_content = bytearray(tensor)
		else:
			#
			# A numeric tensor of some kind
			#
			
			tensor              = np.array(tensor)
			self.dtype          = convert_dtype_np2tf(tensor.dtype)
			self.tensor_shape   = convert_shape_np2tf(tensor.shape, dimNames)
			self.version_number = 0
			self.tensor_content = tensor.tobytes('C')
		
	
	def asByteArray(self):
		b = bytearray()
		if   hasattr(self, "dtype"):                b += enc_tagvalue("enum",    1, self.dtype)
		if   hasattr(self, "tensor_shape"):         b += enc_tagvalue("message", 2, self.tensor_shape)
		if   hasattr(self, "version_number"):       b += enc_tagvalue("int32",   3, self.version_number)
		if   hasattr(self, "tensor_content"):       b += enc_tagvalue("bytes",   4, self.tensor_content)
		return b
	
	def asValue(self, tag, metadata=None):
		return TfValue(tag, tensor=self, metadata=metadata)

class TfTensorShape       (PebbleMessage):
	def __init__(self, dims):
		self.dim = dims
	
	def asByteArray(self):
		b = bytearray()
		if   hasattr(self, "dim"):
			for d in self.dim:
				b                                     += enc_tagvalue("message", 2, d)
		if   hasattr(self, "unknown_rank"):         b += enc_tagvalue("bool",    3, self.unknown_rank)
		return b

class TfDim               (PebbleMessage):
	def __init__(self, size, name=None):
		self.size = size
		if   name       is not None: self.name = name
	
	def asByteArray(self):
		b = bytearray()
		if   hasattr(self, "size"):                 b += enc_tagvalue("int64",   1, self.size)
		if   hasattr(self, "name"):                 b += enc_tagvalue("string",  2, self.name)
		return b

class TfLayout            (PebbleMessage):
	def __init__(self, category):
		self.version  = 0
		self.category = list(category)
	
	def asByteArray(self):
		b = bytearray()
		if   hasattr(self, "version"):              b += enc_tagvalue("int32",   1, self.version)
		if   hasattr(self, "category"):
			for c in self.category:                 b += enc_tagvalue("message", 2, c)
		return b

class TfCategory          (PebbleMessage):
	def __init__(self, title, chart, closed=False):
		self.title  = str(title)
		self.chart  = list(chart)
		self.closed = bool(closed)
	
	def asByteArray(self):
		b = bytearray()
		if   hasattr(self, "title"):                b += enc_tagvalue("string",  1, self.title)
		if   hasattr(self, "chart"):
			for c in self.chart:                    b += enc_tagvalue("message", 2, c)
		if   hasattr(self, "closed"):               b += enc_tagvalue("bool",    3, self.closed)
		return b

class TfChart             (PebbleMessage):
	def __init__(self, title, tags):
		self.title    = str(title)
		self.tag      = list(tags)
	
	def asByteArray(self):
		b = bytearray()
		if   hasattr(self, "title"):                b += enc_tagvalue("string",  1, self.title)
		if   hasattr(self, "tag"):
			for t in self.tag:                      b += enc_tagvalue("string",  2, t)
		return b


def convert_dtype_np2tf(dtype):
	"""Convert a numpy array's dtype to the enum DataType code used by TF."""
	
	dtype  = str(dtype)
	dtype  = {
		"invalid":     TfDataType.INVALID,
		"float32":     TfDataType.FLOAT,
		"float64":     TfDataType.DOUBLE,
		"int32":       TfDataType.INT32,
		"uint8":       TfDataType.UINT8,
		"int16":       TfDataType.INT16,
		"int8":        TfDataType.INT8,
		"string":      TfDataType.STRING,
		"complex64":   TfDataType.COMPLEX64,
		"int64":       TfDataType.INT64,
		"bool":        TfDataType.BOOL,
		# 11-16: Quantized datatypes that don't exist in numpy...
		"uint16":      TfDataType.UINT16,
		"complex128":  TfDataType.COMPLEX128,
		"float16":     TfDataType.HALF,
		# 20-21: TF-specific datatypes...
		"uint32":      TfDataType.UINT32,
		"uint64":      TfDataType.UINT64,
	}[dtype]
	return dtype

def convert_dims_np2tf (xShape, dimNames=None):
	if dimNames is not None:
		assert len(dimNames) == len(xShape)
		dimNames = [str(x) for x in dimNames]
		return [TfDim(size, name) for size, name in zip(xShape, dimNames)]
	else:
		return [TfDim(size)       for size       in xShape]

def convert_shape_np2tf(xShape, dimNames=None):
	return TfTensorShape(convert_dims_np2tf(xShape, dimNames))

def convert_metadata   (displayName        = None,
                        summaryDescription = None,
                        pluginName         = None,
                        content            = None):
	pluginData = None if (pluginName is None and content is None) else TfPluginData(pluginName, content)
	metadata   = None if (displayName is None and summaryDescription is None and pluginData is None) else \
	             TfSummaryMetadata(displayName, summaryDescription, pluginData)
	return metadata


"""
A tiny module to emit TFEvents files.

TFEvents files are a concatenation of "records":

	Record:
		uint64_t dataLen                // Little-Endian
		uint32_t dataLen_maskCRC32C     // Little-Endian
		uint8_t  data[dataLen]
		uint32_t data_maskCRC32C        // Little-Endian
	(repeat unlimited number of times)


The masked CRC32C is defined in Java as

	private static long maskedCRC32(byte[] data){
		crc32.reset();
		crc32.update(data, 0, data.length);
		long x = u32(crc32.getValue());
		return u32(((x >> 15) | u32(x << 17)) + 0xa282ead8);
	}

where the CRC32C is initialized at ~0, uses the Castagnoli polynomial
and is finalized by bit-reversal.


The payload of a Record is a single protobuf Event, defined with all
submessage below (shamelessly ripped off from TensorFlow Github). The first
Record of a file must always contain Event(file_Version="brain.Event:2")


The TFEvents protocol is defined by the contents of the TensorFlow .proto files
below, organized by the hierarchy of their inclusion, namely:

	https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/util/event.proto
		https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/summary.proto
			https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/tensor.proto
				https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/resource_handle.proto
				https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/tensor_shape.proto
				https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/types.proto
	https://github.com/tensorflow/tensorboard/blob/master/tensorboard/plugins/custom_scalar/layout.proto

The entities that exist in those files are below, organized by the hierarchy of
their inclusion (irrelevant/obsolete members removed):

	Event
		double  wall_time
		int         step
		string      file_version
		LogMessage  log_message
			enum        level
				UNKNOWN:     0
				DEBUGGING:  10
				INFO:       20
				WARN:       30
				ERROR:      40
				FATAL:      50
			string      message
		SessionLog  session_log
			enum        status
				UNSPECIFIED: 0
				START:       1
				STOP:        2
				CHECKPOINT:  3
			string      checkpoint_path
			string      msg
		Summary     summary
			repeated Value value
				string           tag
				SummaryMetadata  metadata
					PluginData
						string plugin_name
						bytes  content
					string display_name
					string summary_description
				float            simple_value
				Image            image
					int   height
					int   width
					int   colorspace
						grayscale:        1
						grayscale+alpha:  2
						RGB:              3
						RGBA:             4
						DIGITAL_YUV:      5
						BGRA:             6
					bytes data
				HistogramProto   histo
					double           min
					double           max
					double           num
					double           sum
					repeated packed double sum_squares
					repeated packed double bucket_limit
				Audio            audio
					float            sample_rate   // In Hz
					int              num_channels
					int              length_franes
					bytes            data
					string           content_type
				TensorProto      tensor
					enum DataType    dtype
						DT_INVALID = 0;// Not a legal value for DataType.  Used to indicate a DataType field has not been set.
						DT_FLOAT = 1;  // Data types that all computation devices are expected to be
						DT_DOUBLE = 2; // capable to support.
						DT_INT32 = 3;
						DT_UINT8 = 4;
						DT_INT16 = 5;
						DT_INT8 = 6;
						DT_STRING = 7;
						DT_COMPLEX64 = 8;  // Single-precision complex
						DT_INT64 = 9;
						DT_BOOL = 10;
						DT_QINT8 = 11;     // Quantized int8
						DT_QUINT8 = 12;    // Quantized uint8
						DT_QINT32 = 13;    // Quantized int32
						DT_BFLOAT16 = 14;  // Float32 truncated to 16 bits.  Only for cast ops.
						DT_QINT16 = 15;    // Quantized int16
						DT_QUINT16 = 16;   // Quantized uint16
						DT_UINT16 = 17;
						DT_COMPLEX128 = 18;  // Double-precision complex
						DT_HALF = 19;
						DT_RESOURCE = 20;
						DT_VARIANT = 21;  // Arbitrary C++ data types
						DT_UINT32 = 22;
						DT_UINT64 = 23;
					TensorShapeProto tensor_shape
						repeated Dim dim
							int    size
							string name
						bool unknown_rank // == 0 for all cases of concern to us
					int          version_number   // == 0
					bytes        tensor_content   // Row-major (C-contiguous) order
	
	# Custom Scalars plugin (custom_scalars):
	Layout            layout
		int32             version
		repeated Category category
			string            title
			bool              closed
			repeated Chart    chart
				string            title
				repeated string   tag







// Protocol buffer representing an event that happened during
// the execution of a Brain model.
message Event {
  // Timestamp of the event.
  double wall_time = 1;

  // Global step of the event.
  int64 step = 2;

  oneof what {
    // An event file was started, with the specified version.
    // This is use to identify the contents of the record IO files
    // easily.  Current version is "brain.Event:2".  All versions
    // start with "brain.Event:".
    string file_version = 3;
    // An encoded version of a GraphDef.
    bytes graph_def = 4;
    // A summary was generated.
    Summary summary = 5;
    // The user output a log message. Not all messages are logged, only ones
    // generated via the Python tensorboard_logging module.
    LogMessage log_message = 6;
    // The state of the session which can be used for restarting after crashes.
    SessionLog session_log = 7;
    // The metadata returned by running a session.run() call.
    TaggedRunMetadata tagged_run_metadata = 8;
    // An encoded version of a MetaGraphDef.
    bytes meta_graph_def = 9;
  }
}

// Protocol buffer used for logging messages to the events file.
message LogMessage {
  enum Level {
    UNKNOWN = 0;
    // Note: The logging level 10 cannot be named DEBUG. Some software
    // projects compile their C/C++ code with -DDEBUG in debug builds. So the
    // C++ code generated from this file should not have an identifier named
    // DEBUG.
    DEBUGGING = 10;
    INFO = 20;
    WARN = 30;
    ERROR = 40;
    FATAL = 50;
  }
  Level level = 1;
  string message = 2;
}

// Protocol buffer used for logging session state.
message SessionLog {
  enum SessionStatus {
    STATUS_UNSPECIFIED = 0;
    START = 1;
    STOP = 2;
    CHECKPOINT = 3;
  }

  SessionStatus status = 1;
  // This checkpoint_path contains both the path and filename.
  string checkpoint_path = 2;
  string msg = 3;
}

// For logging the metadata output for a single session.run() call.
message TaggedRunMetadata {
  // Tag name associated with this metadata.
  string tag = 1;
  // Byte-encoded version of the `RunMetadata` proto in order to allow lazy
  // deserialization.
  bytes run_metadata = 2;
}


// Metadata associated with a series of Summary data
message SummaryDescription {
  // Hint on how plugins should process the data in this series.
  // Supported values include "scalar", "histogram", "image", "audio"
  string type_hint = 1;
}

// Serialization format for histogram module in
// core/lib/histogram/histogram.h
message HistogramProto {
  double min = 1;
  double max = 2;
  double num = 3;
  double sum = 4;
  double sum_squares = 5;

  // Parallel arrays encoding the bucket boundaries and the bucket values.
  // bucket(i) is the count for the bucket i.  The range for
  // a bucket is:
  //   i == 0:  -DBL_MAX .. bucket_limit(0)
  //   i != 0:  bucket_limit(i-1) .. bucket_limit(i)
  repeated double bucket_limit = 6 [packed = true];
  repeated double bucket = 7 [packed = true];
};

// A SummaryMetadata encapsulates information on which plugins are able to make
// use of a certain summary value.
message SummaryMetadata {
  message PluginData {
    // The name of the plugin this data pertains to.
    string plugin_name = 1;

    // The content to store for the plugin. The best practice is for this to be
    // a binary serialized protocol buffer.
    bytes content = 2;
  }

  // Data that associates a summary with a certain plugin.
  PluginData plugin_data = 1;

  // Display name for viewing in TensorBoard.
  string display_name = 2;

  // Longform readable description of the summary sequence. Markdown supported.
  string summary_description = 3;
};

// A Summary is a set of named values to be displayed by the
// visualizer.
//
// Summaries are produced regularly during training, as controlled by
// the "summary_interval_secs" attribute of the training operation.
// Summaries are also produced at the end of an evaluation.
message Summary {
  message Image {
    // Dimensions of the image.
    int32 height = 1;
    int32 width = 2;
    // Valid colorspace values are
    //   1 - grayscale
    //   2 - grayscale + alpha
    //   3 - RGB
    //   4 - RGBA
    //   5 - DIGITAL_YUV
    //   6 - BGRA
    int32 colorspace = 3;
    // Image data in encoded format.  All image formats supported by
    // image_codec::CoderUtil can be stored here.
    bytes encoded_image_string = 4;
  }

  message Audio {
    // Sample rate of the audio in Hz.
    float sample_rate = 1;
    // Number of channels of audio.
    int64 num_channels = 2;
    // Length of the audio in frames (samples per channel).
    int64 length_frames = 3;
    // Encoded audio data and its associated RFC 2045 content type (e.g.
    // "audio/wav").
    bytes encoded_audio_string = 4;
    string content_type = 5;
  }

  message Value {
    // This field is deprecated and will not be set.
    string node_name = 7;

    // Tag name for the data. Used by TensorBoard plugins to organize data. Tags
    // are often organized by scope (which contains slashes to convey
    // hierarchy). For example: foo/bar/0
    string tag = 1;

    // Contains metadata on the summary value such as which plugins may use it.
    // Take note that many summary values may lack a metadata field. This is
    // because the FileWriter only keeps a metadata object on the first summary
    // value with a certain tag for each tag. TensorBoard then remembers which
    // tags are associated with which plugins. This saves space.
    SummaryMetadata metadata = 9;

    // Value associated with the tag.
    oneof value {
      float simple_value = 2;
      bytes obsolete_old_style_histogram = 3;
      Image image = 4;
      HistogramProto histo = 5;
      Audio audio = 6;
      TensorProto tensor = 8;
    }
  }

  // Set of values for the summary.
  repeated Value value = 1;
}

// Protocol buffer representing a tensor.
message TensorProto {
  DataType dtype = 1;

  // Shape of the tensor.  TODO(touts): sort out the 0-rank issues.
  TensorShapeProto tensor_shape = 2;

  // Only one of the representations below is set, one of "tensor_contents" and
  // the "xxx_val" attributes.  We are not using oneof because as oneofs cannot
  // contain repeated fields it would require another extra set of messages.

  // Version number.
  //
  // In version 0, if the "repeated xxx" representations contain only one
  // element, that element is repeated to fill the shape.  This makes it easy
  // to represent a constant Tensor with a single value.
  int32 version_number = 3;

  // Serialized raw tensor content from either Tensor::AsProtoTensorContent or
  // memcpy in tensorflow::grpc::EncodeTensorToByteBuffer. This representation
  // can be used for all tensor types. The purpose of this representation is to
  // reduce serialization overhead during RPC call by avoiding serialization of
  // many repeated small items.
  bytes tensor_content = 4;

  // Type specific representations that make it easy to create tensor protos in
  // all languages.  Only the representation corresponding to "dtype" can
  // be set.  The values hold the flattened representation of the tensor in
  // row major order.

  // DT_HALF. Note that since protobuf has no int16 type, we'll have some
  // pointless zero padding for each value here.
  repeated int32 half_val = 13 [packed = true];

  // DT_FLOAT.
  repeated float float_val = 5 [packed = true];

  // DT_DOUBLE.
  repeated double double_val = 6 [packed = true];

  // DT_INT32, DT_INT16, DT_INT8, DT_UINT8.
  repeated int32 int_val = 7 [packed = true];

  // DT_STRING
  repeated bytes string_val = 8;

  // DT_COMPLEX64. scomplex_val(2*i) and scomplex_val(2*i+1) are real
  // and imaginary parts of i-th single precision complex.
  repeated float scomplex_val = 9 [packed = true];

  // DT_INT64
  repeated int64 int64_val = 10 [packed = true];

  // DT_BOOL
  repeated bool bool_val = 11 [packed = true];

  // DT_COMPLEX128. dcomplex_val(2*i) and dcomplex_val(2*i+1) are real
  // and imaginary parts of i-th double precision complex.
  repeated double dcomplex_val = 12 [packed = true];

  // DT_RESOURCE
  repeated ResourceHandleProto resource_handle_val = 14;

  // DT_VARIANT
  repeated VariantTensorDataProto variant_val = 15;

  // DT_UINT32
  repeated uint32 uint32_val = 16 [packed = true];

  // DT_UINT64
  repeated uint64 uint64_val = 17 [packed = true];
};

// Protocol buffer representing the serialization format of DT_VARIANT tensors.
message VariantTensorDataProto {
  // Name of the type of objects being serialized.
  string type_name = 1;
  // Portions of the object that are not Tensors.
  bytes metadata = 2;
  // Tensors contained within objects being serialized.
  repeated TensorProto tensors = 3;
}

// Protocol buffer representing a handle to a tensorflow resource. Handles are
// not valid across executions, but can be serialized back and forth from within
// a single run.
message ResourceHandleProto {
  // Unique name for the device containing the resource.
  string device = 1;

  // Container in which this resource is placed.
  string container = 2;

  // Unique name of this resource.
  string name = 3;

  // Hash code for the type of the resource. Is only valid in the same device
  // and in the same execution.
  uint64 hash_code = 4;

  // For debug-only, the name of the type pointed to by this handle, if
  // available.
  string maybe_type_name = 5;
};

// Dimensions of a tensor.
message TensorShapeProto {
  // One dimension of the tensor.
  message Dim {
    // Size of the tensor in that dimension.
    // This value must be >= -1, but values of -1 are reserved for "unknown"
    // shapes (values of -1 mean "unknown" dimension).  Certain wrappers
    // that work with TensorShapeProto may fail at runtime when deserializing
    // a TensorShapeProto containing a dim value of -1.
    int64 size = 1;

    // Optional name of the tensor dimension.
    string name = 2;
  };

  // Dimensions of the tensor, such as {"input", 30}, {"output", 40}
  // for a 30 x 40 2D tensor.  If an entry has size -1, this
  // corresponds to a dimension of unknown size. The names are
  // optional.
  //
  // The order of entries in "dim" matters: It indicates the layout of the
  // values in the tensor in-memory representation.
  //
  // The first entry in "dim" is the outermost dimension used to layout the
  // values, the last entry is the innermost dimension.  This matches the
  // in-memory layout of RowMajor Eigen tensors.
  //
  // If "dim.size()" > 0, "unknown_rank" must be false.
  repeated Dim dim = 2;

  // If true, the number of dimensions in the shape is unknown.
  //
  // If true, "dim.size()" must be 0.
  bool unknown_rank = 3;
};

// LINT.IfChange
enum DataType {
  // Not a legal value for DataType.  Used to indicate a DataType field
  // has not been set.
  DT_INVALID = 0;

  // Data types that all computation devices are expected to be
  // capable to support.
  DT_FLOAT = 1;
  DT_DOUBLE = 2;
  DT_INT32 = 3;
  DT_UINT8 = 4;
  DT_INT16 = 5;
  DT_INT8 = 6;
  DT_STRING = 7;
  DT_COMPLEX64 = 8;  // Single-precision complex
  DT_INT64 = 9;
  DT_BOOL = 10;
  DT_QINT8 = 11;     // Quantized int8
  DT_QUINT8 = 12;    // Quantized uint8
  DT_QINT32 = 13;    // Quantized int32
  DT_BFLOAT16 = 14;  // Float32 truncated to 16 bits.  Only for cast ops.
  DT_QINT16 = 15;    // Quantized int16
  DT_QUINT16 = 16;   // Quantized uint16
  DT_UINT16 = 17;
  DT_COMPLEX128 = 18;  // Double-precision complex
  DT_HALF = 19;
  DT_RESOURCE = 20;
  DT_VARIANT = 21;  // Arbitrary C++ data types
  DT_UINT32 = 22;
  DT_UINT64 = 23;

  // Do not use!  These are only for parameters.  Every enum above
  // should have a corresponding value below (verified by types_test).
  DT_FLOAT_REF = 101;
  DT_DOUBLE_REF = 102;
  DT_INT32_REF = 103;
  DT_UINT8_REF = 104;
  DT_INT16_REF = 105;
  DT_INT8_REF = 106;
  DT_STRING_REF = 107;
  DT_COMPLEX64_REF = 108;
  DT_INT64_REF = 109;
  DT_BOOL_REF = 110;
  DT_QINT8_REF = 111;
  DT_QUINT8_REF = 112;
  DT_QINT32_REF = 113;
  DT_BFLOAT16_REF = 114;
  DT_QINT16_REF = 115;
  DT_QUINT16_REF = 116;
  DT_UINT16_REF = 117;
  DT_COMPLEX128_REF = 118;
  DT_HALF_REF = 119;
  DT_RESOURCE_REF = 120;
  DT_VARIANT_REF = 121;
  DT_UINT32_REF = 122;
  DT_UINT64_REF = 123;
}

/**
 * Encapsulates information on a single chart. Many charts appear in a category.
 */
message Chart {
  // The title shown atop this chart. This field is optional and defaults to a
  // comma-separated list of tag regular expressions.
  string title = 1;

  // A list of regular expressions for tags that should appear in this chart.
  // Tags are matched based from beginning to end.
  repeated string tag = 2;
}

/**
 * A category contains a group of charts. Each category maps to a collapsible
 * within the dashboard.
 */
message Category {
  // This string appears atop each grouping of charts within the dashboard.
  string title = 1;

  // Encapsulates data on charts to be shown in the category.
  repeated Chart chart = 2;

  // Whether this category should be initially closed. False by default.
  bool closed = 3;
}

/**
 * A layout encapsulates how charts are laid out within the custom scalars
 * dashboard.
 */
message Layout {
  // Version `0` is the only supported version.
  int32 version = 1;

  // The categories here are rendered from top to bottom.
  repeated Category category = 2;
}








TO BE LOOKED AT:
	https://github.com/tensorflow/tensorboard/blob/master/tensorboard/plugins/projector/projector_config.proto
	https://github.com/tensorflow/tensorboard/blob/master/tensorboard/plugins/pr_curve
"""
