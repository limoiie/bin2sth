import idaapi
# from idaapi import *
import idc
# from idc import *
import idautils

import os


def is_be(info):
  try:
    return info.is_be()
  except:
    return idaapi.cvar.inf.mf


def functions(seg):
  return idautils.Functions(idc.SegStart(seg), idc.SegEnd(seg))


def all_functions():
  for seg in idautils.Segments():
    for fun in functions(seg):
      yield seg, fun


def is_api(func_ea):
  flags = idc.GetFunctionFlags(func_ea)
  return flags & idc.FUNC_LIB or flags & idc.FUNC_THUNK


def filepath():
  idb_path = idc.GetIdbPath()
  filepath = os.path.splitext(idb_path)
  return filepath[0]


def to_hex_(i):
  return hex(i)[2:-1].upper()


def to_hex(i):
  return hex(i)[:2] + to_hex_(i)


def to_hex_inv_(i):
  return to_hex_(i)[::-1]


def to_hex_inv(i):
  return hex(i)[:2] + to_hex_inv_(i)
