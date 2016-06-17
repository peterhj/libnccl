#![allow(non_camel_case_types)]

extern crate cuda;
extern crate libc;

use cuda::ffi::runtime::{cudaStream_t};
use libc::*;

pub enum ncclComm {}
pub type ncclComm_t = *mut ncclComm;

pub const NCCL_UNIQUE_ID_BYTES: usize = 128;

#[repr(C)]
pub struct ncclUniqueId {
  pub internal: [c_char; NCCL_UNIQUE_ID_BYTES],
}

impl Clone for ncclUniqueId {
  fn clone(&self) -> ncclUniqueId {
    let mut new_id = ncclUniqueId{internal: [0; NCCL_UNIQUE_ID_BYTES]};
    new_id.internal.copy_from_slice(&self.internal);
    new_id
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(C)]
pub enum ncclResult_t {
  Success                   = 0,
  UnhandledCudaError        = 1,
  SystemError               = 2,
  InternalError             = 3,
  InvalidDeviceError        = 4,
  InvalidRank               = 5,
  UnsupportedDeviceCount    = 6,
  DeviceNotFound            = 7,
  InvalidDeviceIndex        = 8,
  LibWrapperNotSet          = 9,
  CudaMallocFailed          = 10,
  RankMismatch              = 11,
  InvalidArgument           = 12,
  InvalidType               = 13,
  InvalidOperation          = 14,
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub enum ncclRedOp_t {
  Sum   = 0,
  Prod  = 1,
  Max   = 2,
  Min   = 3,
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub enum ncclDataType_t {
  Char      = 0,
  Int       = 1,
  Half      = 2,
  Float     = 3,
  Double    = 4,
  Int64     = 5,
  Uint64    = 6,
}

#[link(name = "nccl_native", kind = "static")]
extern "C" {
  pub fn ncclGetUniqueId(uniqueId: *mut ncclUniqueId) -> ncclResult_t;
  pub fn ncclCommInitRank(comm: *mut ncclComm_t, ndev: c_int, commId: ncclUniqueId, rank: c_int) -> ncclResult_t;
  pub fn ncclCommInitAll(comm: *mut ncclComm_t, ndev: c_int, devlist: *mut c_int) -> ncclResult_t;
  pub fn ncclCommDestroy(comm: ncclComm_t);
  pub fn ncclGetErrorString(result: ncclResult_t) -> *const c_char;
  pub fn ncclCommCount(comm: ncclComm_t, count: *mut c_int) -> ncclResult_t;
  pub fn ncclCommCuDevice(comm: ncclComm_t, device: *mut c_int) -> ncclResult_t;
  pub fn ncclCommUserRank(comm: ncclComm_t, rank: *mut c_int) -> ncclResult_t;
  pub fn ncclReduce(sendbuf: *const c_void, recvbuf: *mut c_void, count: c_int, datatype: ncclDataType_t, op: ncclRedOp_t, root: c_int, comm: ncclComm_t, stream: cudaStream_t) -> ncclResult_t;
  pub fn ncclAllReduce(sendbuf: *const c_void, recvbuf: *mut c_void, count: c_int, datatype: ncclDataType_t, op: ncclRedOp_t, comm: ncclComm_t, stream: cudaStream_t) -> ncclResult_t;
  pub fn ncclReduceScatter(sendbuf: *const c_void, recvbuf: *mut c_void, recvcount: c_int, datatype: ncclDataType_t, op: ncclRedOp_t, comm: ncclComm_t, stream: cudaStream_t) -> ncclResult_t;
  pub fn ncclBcast(buf: *mut c_void, count: c_int, datatype: ncclDataType_t, root: c_int, comm: ncclComm_t, stream: cudaStream_t) -> ncclResult_t;
  pub fn ncclAllGather(sendbuf: *const c_void, count: c_int, datatype: ncclDataType_t, recvbuf: *mut c_void, comm: ncclComm_t, stream: cudaStream_t) -> ncclResult_t;
}
