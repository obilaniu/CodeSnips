#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
A tiny module to emit TFEvents files.

TFEvents files are a concatenation of "records":

	Record:
		uint64_t dataLen
		uint32_t dataLen_maskCRC32C
		uint8_t  data[dataLen]
		uint32_t data_maskCRC32C
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
    string content = 2;
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

  // TODO(josh11b): DT_GENERIC_PROTO = ??;
  // TODO(jeff,josh11b): DT_UINT64?  DT_UINT32?

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
}
// LINT.ThenChange(https://www.tensorflow.org/code/tensorflow/c/c_api.h,https://www.tensorflow.org/code/tensorflow/go/tensor.go)

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
"""


import crc32c, numpy as np, struct, time



#
# Custom TF CRC32C checksum variant
#
# The only difference is in the finalization function, apparently.
#

class TFCRC32C(crc32c.CRC32C):
	def finalize(self):
		v = super(TFCRC32C, self).finalize()
		return ((v>>15 | v<<17) + 0xa282ead8) & 0xFFFFFFFF



#
# TF Protocol
#

# Wire encoders
def enc_uint64(i):
	i   = long(i)
	i  &= 0xFFFFFFFFFFFFFFFF
	b   = bytearray([i & 0x7F])
	i >>= 7
	
	while i:
		b[-1] |= 0x80
		b.append(i & 0x7F)
		i >>= 7
	
	return b
def enc_int64(i):    return enc_uint64(i)
def enc_uint32(i):   return enc_uint64(long(i) & 0xFFFFFFFF)
def enc_int32(i):    return enc_uint64(i)
def enc_sint64(i):
	i   = long(i)
	i  &= 0xFFFFFFFFFFFFFFFF
	i <<= 1
	i  ^= -(i>>64)
	i  &= 0xFFFFFFFFFFFFFFFF
	return enc_uint64(i)
def enc_sint32(i):
	i   = long(i)
	i  &= 0xFFFFFFFF
	i <<= 1
	i  ^= -(i>>32)
	i  &= 0xFFFFFFFF
	return enc_uint32(i)
def enc_bool(b):     return bytearray([1 if b else 0])
def enc_enum(e):     return enc_int32(e)
def enc_fixed64(i):  return bytearray(struct.pack("<Q", long (i)))
def enc_sfixed64(i): return bytearray(struct.pack("<q", long (i)))
def enc_float64(i):  return bytearray(struct.pack("<d", float(i)))
def enc_fixed32(i):  return bytearray(struct.pack("<I", long (i)))
def enc_sfixed32(i): return bytearray(struct.pack("<i", long (i)))
def enc_float32(i):  return bytearray(struct.pack("<f", float(i)))
def enc_tag(tag, wire):
	assert isinstance(tag,  (int, long))      and \
	       isinstance(wire, (int, long))      and \
	       tag >= 0                           and \
	       tag <  2**29-1                     and \
	       not (tag >= 10000 and tag < 19999) and \
	       wire in [0,1,2,5]
	
	return enc_uint32(tag << 3 | wire)
def enc_delimited(buf):
	assert isinstance(buf, bytearray)
	return enc_uint64(len(buf)) + buf
def enc_tagvalue(tagtype, tag, val, required=False):
	assert isinstance(tagtype, str)
	
	if   tagtype == "double":    return enc_tag(tag, 1)+enc_float64(val)   if(val != 0 or required) else bytearray()
	elif tagtype == "float":     return enc_tag(tag, 5)+enc_float32(val)   if(val != 0 or required) else bytearray()
	elif tagtype == "int32":     return enc_tag(tag, 0)+enc_int32(val)     if(val != 0 or required) else bytearray()
	elif tagtype == "int64":     return enc_tag(tag, 0)+enc_int64(val)     if(val != 0 or required) else bytearray()
	elif tagtype == "uint32":    return enc_tag(tag, 0)+enc_uint32(val)    if(val != 0 or required) else bytearray()
	elif tagtype == "uint64":    return enc_tag(tag, 0)+enc_uint64(val)    if(val != 0 or required) else bytearray()
	elif tagtype == "sint32":    return enc_tag(tag, 0)+enc_sint32(val)    if(val != 0 or required) else bytearray()
	elif tagtype == "sint64":    return enc_tag(tag, 0)+enc_sint64(val)    if(val != 0 or required) else bytearray()
	elif tagtype == "fixed32":   return enc_tag(tag, 5)+enc_fixed32(val)   if(val != 0 or required) else bytearray()
	elif tagtype == "fixed64":   return enc_tag(tag, 1)+enc_fixed64(val)   if(val != 0 or required) else bytearray()
	elif tagtype == "sfixed32":  return enc_tag(tag, 5)+enc_sfixed32(val)  if(val != 0 or required) else bytearray()
	elif tagtype == "sfixed64":  return enc_tag(tag, 1)+enc_sfixed64(val)  if(val != 0 or required) else bytearray()
	elif tagtype == "bool":      return enc_tag(tag, 0)+enc_bool(val)      if(val      or required) else bytearray()
	elif tagtype == "enum":      return enc_tag(tag, 0)+enc_enum(val)      if(val != 0 or required) else bytearray()
	elif tagtype == "string":
		val = bytearray(val);    return enc_tag(tag, 2)+enc_delimited(val) if(len(val) or required) else bytearray()
	elif tagtype == "bytes"   or \
	     tagtype == "packed"  or \
	     tagtype == "message":   return enc_tag(tag, 2)+enc_delimited(val) if(len(val) or required) else bytearray()
	else:
		raise ValueError("Illegal tag type \""+tagtype+"\"!")


# Message Hierarchy
class Event(object):
	@property
	def asbytearray(self):
		b = bytearray()
		if   hasattr(self, "wall_time"):            b += enc_tagvalue("double",  1, self.wall_time)
		if   hasattr(self, "step"):                 b += enc_tagvalue("int64",   2, self.step)
		
		if   hasattr(self, "file_version"):         b += enc_tagvalue("string",  3, self.file_version)
		elif hasattr(self, "graph_def"):            b += enc_tagvalue("bytes",   4, self.graph_def)
		elif hasattr(self, "summary"):              b += enc_tagvalue("message", 5, self.summary.asbytearray)
		elif hasattr(self, "log_message"):          b += enc_tagvalue("message", 6, self.log_message.asbytearray)
		elif hasattr(self, "session_log"):          b += enc_tagvalue("message", 7, self.session_log.asbytearray)
		elif hasattr(self, "tagged_run_metadata"):  b += enc_tagvalue("message", 8, self.tagged_run_metadata.asbytearray)
		elif hasattr(self, "meta_graph_def"):       b += enc_tagvalue("bytes",   9, self.meta_graph_def)
		else: raise ValueError("The event is empty!")
		
		return b
class LogMessage(object):
	@property
	def asbytearray(self):
		b = bytearray()
		if   hasattr(self, "level"):                b += enc_tagvalue("enum",    1, self.level)
		if   hasattr(self, "message"):              b += enc_tagvalue("string",  2, self.message)
		return b
class SessionLog(object):
	@property
	def asbytearray(self):
		b = bytearray()
		if   hasattr(self, "status"):               b += enc_tagvalue("enum",    1, self.status)
		if   hasattr(self, "checkpoint_path"):      b += enc_tagvalue("string",  2, self.checkpoint_path)
		if   hasattr(self, "msg"):                  b += enc_tagvalue("string",  3, self.msg)
		return b
class TaggedRunMetadata(object):
	@property
	def asbytearray(self):
		b = bytearray()
		if   hasattr(self, "tag"):                  b += enc_tagvalue("string",  1, self.tag)
		if   hasattr(self, "run_metadata"):         b += enc_tagvalue("bytes",   2, self.run_metadata)
		return b
class SummaryDescription(object):
	@property
	def asbytearray(self):
		b = bytearray()
		if   hasattr(self, "type_hint"):            b += enc_tagvalue("string",  1, self.type_hint)
		return b
class HistogramProto(object):
	@property
	def asbytearray(self):
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
class SummaryMetadata(object):
	@property
	def asbytearray(self):
		b = bytearray()
		if   hasattr(self, "plugin_data"):          b += enc_tagvalue("message", 1, self.plugin_data.asbytearray)
		if   hasattr(self, "display_name"):         b += enc_tagvalue("string",  2, self.display_name)
		if   hasattr(self, "summary_description"):  b += enc_tagvalue("string",  3, self.summary_description)
		return b
class PluginData(object):
	@property
	def asbytearray(self):
		b = bytearray()
		if   hasattr(self, "plugin_name"):          b += enc_tagvalue("string",  1, self.plugin_name)
		if   hasattr(self, "content"):              b += enc_tagvalue("string",  2, self.content)
		return b
class Summary(object):
	@property
	def asbytearray(self):
		b = bytearray()
		if   hasattr(self, "value"):
			for v in self.value:                    b += enc_tagvalue("message", 1, v.asbytearray)
		return b
class Image(object):
	@property
	def asbytearray(self):
		b = bytearray()
		if   hasattr(self, "height"):               b += enc_tagvalue("int32",   1, self.height)
		if   hasattr(self, "width"):                b += enc_tagvalue("int32",   2, self.width)
		if   hasattr(self, "colorspace"):           b += enc_tagvalue("int32",   3, self.colorspace)
		if   hasattr(self, "encoded_image_string"): b += enc_tagvalue("bytes",   4, self.encoded_image_string)
		return b
class Audio(object):
	@property
	def asbytearray(self):
		b = bytearray()
		if   hasattr(self, "sample_rate"):          b += enc_tagvalue("float",   1, self.sample_rate)
		if   hasattr(self, "num_channels"):         b += enc_tagvalue("int64",   2, self.num_channels)
		if   hasattr(self, "length_frames"):        b += enc_tagvalue("int64",   3, self.length_frames)
		if   hasattr(self, "encoded_audio_string"): b += enc_tagvalue("bytes",   4, self.encoded_audio_string)
		if   hasattr(self, "content_type"):         b += enc_tagvalue("string",  5, self.content_type)
		return b
class Value(object):
	@property
	def asbytearray(self):
		b = bytearray()
		if   hasattr(self, "tag"):                  b += enc_tagvalue("string",  1, self.tag)
		if   hasattr(self, "metadata"):             b += enc_tagvalue("message", 9, self.metadata.asbytearray)
		
		if   hasattr(self, "simple_value"):         b += enc_tagvalue("float",   2, self.simple_value)
		elif hasattr(self, "image"):                b += enc_tagvalue("message", 4, self.image.asbytearray)
		elif hasattr(self, "histo"):                b += enc_tagvalue("message", 5, self.histo.asbytearray)
		elif hasattr(self, "audio"):                b += enc_tagvalue("message", 6, self.audio.asbytearray)
		elif hasattr(self, "tensor"):               b += enc_tagvalue("message", 8, self.tensor.asbytearray)
		
		return b
class TensorProto(object):
	@property
	def asbytearray(self):
		b = bytearray()
		if   hasattr(self, "dtype"):                b += enc_tagvalue("enum",    1, self.dtype)
		if   hasattr(self, "tensor_shape"):         b += enc_tagvalue("message", 2, self.tensor_shape.asbytearray)
		if   hasattr(self, "version_number"):       b += enc_tagvalue("int32",   3, self.version_number)
		if   hasattr(self, "tensor_content"):       b += enc_tagvalue("bytes",   4, self.tensor_content)
		return b
class VariantTensorDataProto(object):
	@property
	def asbytearray(self):
		b = bytearray()
		if   hasattr(self, "name"):                 b += enc_tagvalue("string",  1, self.name)
		if   hasattr(self, "metadata"):             b += enc_tagvalue("bytes",   2, self.metadata)
		if   hasattr(self, "tensors"):
			for t in self.tensors:
				b                                     += enc_tagvalue("message", 3, t.asbytearray)
		return b
class TensorShapeProto(object):
	@property
	def asbytearray(self):
		b = bytearray()
		if   hasattr(self, "dim"):
			for d in self.dim:
				b                                     += enc_tagvalue("message", 2, d.asbytearray)
		if   hasattr(self, "unknown_rank"):         b += enc_tagvalue("bool",    3, self.unknown_rank)
		return b
class Dim(object):
	@property
	def asbytearray(self):
		b = bytearray()
		if   hasattr(self, "size"):                 b += enc_tagvalue("int64",   1, self.size)
		if   hasattr(self, "name"):                 b += enc_tagvalue("string",  2, self.name)
		return b
class ResourceHandleProto(object):
	@property
	def asbytearray(self):
		b = bytearray()
		if   hasattr(self, "device"):               b += enc_tagvalue("string",  1, self.device)
		if   hasattr(self, "container"):            b += enc_tagvalue("string",  2, self.container)
		if   hasattr(self, "name"):                 b += enc_tagvalue("string",  3, self.name)
		if   hasattr(self, "hash_code"):            b += enc_tagvalue("uint64",  4, self.hash_code)
		if   hasattr(self, "maybe_type_name"):      b += enc_tagvalue("string",  5, self.maybe_type_name)
		return b



# RecordEmitter
class RecordEmitter(object):
	def __init__(self, stream):
		self.stream = stream
	def emitEvent(self, event):
		data   = event.asbytearray
		header = enc_fixed64(len(data))
		pkt    = header                                            + \
		         enc_fixed32(TFCRC32C().update(header).finalize()) + \
		         data                                              + \
		         enc_fixed32(TFCRC32C().update(data).finalize())
		self.stream.write(pkt)


# Test
if __name__ == "__main__":
	with open("test.tfevents", "ab") as f:
		r = RecordEmitter(f)
		
		e = Event()
		e.wall_time     = time.time()
		e.file_version  = "brain.Event:2"
		
		r.emitEvent(e)
		
		for i in xrange(100):
			a = np.random.normal(0.5, 1.0, size=(30,40)).astype("float64")
			buckets = []
			lim     = 1e-12
			while lim<1e20:
				buckets += [lim]
				lim     *= 1.1
			buckets = [np.finfo("float64").min] + [-l for l in buckets[::-1]] + [0] + buckets + [np.finfo("float64").max]
			
			e = Event()
			e.wall_time     = time.time()
			e.step          = i
			e.summary       = Summary()
			e.summary.value = [Value(), Value()]
			e.summary.value[0].tag          = "e/loss"
			e.summary.value[0].simple_value = 0.69 * 0.95**i
			#e.summary.value[2].tag          = "e/mat"
			#e.summary.value[2].tensor       = TensorProto()
			#e.summary.value[2].tensor.dtype          = 1 # DT_FLOAT
			#e.summary.value[2].tensor.tensor_shape   = TensorShapeProto()
			#e.summary.value[2].tensor.tensor_shape.dim = [Dim(), Dim()]
			#e.summary.value[2].tensor.tensor_shape.dim[0].size = 3
			#e.summary.value[2].tensor.tensor_shape.dim[1].size = 4
			#e.summary.value[2].tensor.version_number = 0
			#e.summary.value[2].tensor.tensor_content = bytearray(a)
			e.summary.value[1].tag          = "e/hist"
			e.summary.value[1].histo        = HistogramProto()
			e.summary.value[1].histo.min_         = float(np.min(a))
			e.summary.value[1].histo.max_         = float(np.max(a))
			e.summary.value[1].histo.num          = float(np.prod(a.shape))
			e.summary.value[1].histo.sum_         = float(np.sum(a))
			e.summary.value[1].histo.sum_squares  = float(np.sum(a**2))
			e.summary.value[1].histo.bucket_limit = buckets[1:]
			e.summary.value[1].histo.bucket       = np.histogram(a, buckets)[0]
			#import pdb;pdb.set_trace()
			
			r.emitEvent(e)

