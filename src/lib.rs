extern crate nccl_sys;

extern crate cuda;
extern crate libc;

use cuda::ffi::runtime::{cudaStream_t};
use nccl_sys::*;

use libc::*;
use std::ptr::{null_mut};

//pub mod ffi;

pub trait NcclDataType {
  fn kind() -> ncclDataType_t;
}

impl NcclDataType for f32 {
  fn kind() -> ncclDataType_t { ncclDataType_t::Float }
}

pub trait NcclOp {
  fn kind() -> ncclRedOp_t;
}

pub struct NcclSumOp;

impl NcclOp for NcclSumOp {
  fn kind() -> ncclRedOp_t { ncclRedOp_t::Sum }
}

#[derive(Clone)]
pub struct NcclUniqueId {
  raw:  ncclUniqueId,
}

impl NcclUniqueId {
  pub fn create() -> Result<NcclUniqueId, ncclResult_t> {
    let mut raw = ncclUniqueId{internal: [0; NCCL_UNIQUE_ID_BYTES]};
    let res = unsafe { ncclGetUniqueId(&mut raw as *mut _) };
    if res != ncclResult_t::Success {
      return Err(res);
    }
    Ok(NcclUniqueId{raw: raw})
  }
}

pub struct NcclComm {
  ptr:  ncclComm_t,
}

impl Drop for NcclComm {
  fn drop(&mut self) {
    unsafe { ncclCommDestroy(self.ptr) };
  }
}

impl NcclComm {
  pub fn create(rank: usize, num_devices: usize, comm_id: NcclUniqueId) -> Result<NcclComm, ncclResult_t> {
    let mut inner: ncclComm_t = null_mut();
    let res = unsafe { ncclCommInitRank(&mut inner as *mut _, num_devices as c_int, comm_id.raw.clone(), rank as c_int) };
    if res != ncclResult_t::Success {
      return Err(res);
    }
    Ok(NcclComm{ptr: inner})
  }

  pub fn rank(&self) -> Result<usize, ncclResult_t> {
    let mut rank = 0;
    let res = unsafe { ncclCommUserRank(self.ptr, &mut rank) };
    if res != ncclResult_t::Success {
      return Err(res);
    }
    Ok(rank as usize)
  }

  pub fn device_idx(&self) -> Result<usize, ncclResult_t> {
    let mut device_idx = 0;
    let res = unsafe { ncclCommCuDevice(self.ptr, &mut device_idx) };
    if res != ncclResult_t::Success {
      return Err(res);
    }
    Ok(device_idx as usize)
  }

  pub fn size(&self) -> Result<usize, ncclResult_t> {
    let mut count = 0;
    let res = unsafe { ncclCommCount(self.ptr, &mut count) };
    if res != ncclResult_t::Success {
      return Err(res);
    }
    Ok(count as usize)
  }

  pub unsafe fn allreduce<T, Op>(&self, src: *const T, len: usize, dst: *mut T, _op: Op, stream: cudaStream_t) -> Result<(), ncclResult_t>
  where T: NcclDataType, Op: NcclOp {
    let res = ncclAllReduce(src as *const _, dst as *mut _, len as c_int, T::kind(), Op::kind(), self.ptr, stream);
    if res != ncclResult_t::Success {
      return Err(res);
    }
    Ok(())
  }
}
