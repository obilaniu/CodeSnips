# -*- coding: utf-8 -*-


#
# Imports
#

import numpy                  as np
import struct
import time

from   pysnips.discrete.cksum.crc import CRC32C



#
# Basic ProtoBuf emitter utilities
#

class TFCRC32C(CRC32C):
	"""
	Custom TF CRC32C checksum variant.
	
	The only difference is in the finalization function.
	"""
	
	def finalize(self):
		"""
		TF does a special "masked" finalization on CRC32C's output. This
		involves a rotate-right by 15 bits and the addition of 0xa282ead8.
		
		What this accomplishes is unknown, aside from decreasing compatibility.
		"""
		
		v = super(TFCRC32C, self).finalize()
		return ((v>>15 | v<<17) + 0xa282ead8) & 0xFFFFFFFF

def tfcrc32c(buf):
	return TFCRC32C().update(buf).finalize()

class PebbleMessage(object):
	"""Base class for a ProtoBuf message."""
	
	def __init__(self, **kwargs):
		self.__dict__.update(kwargs)
	def asByteArray(self):
		return bytearray()


#
# TF Protocol utility functions
#
# Mostly wire data encoders
#

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
	     tagtype == "packed":
		val = bytearray(val);    return enc_tag(tag, 2)+enc_delimited(val) if(len(val) or required) else bytearray()
	elif tagtype == "message":
		val = val.asByteArray(); return enc_tag(tag, 2)+enc_delimited(val) if(len(val) or required) else bytearray()
	else:
		raise ValueError("Illegal tag type \""+tagtype+"\"!")
