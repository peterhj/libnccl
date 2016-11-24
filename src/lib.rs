extern crate cuda;
extern crate float;
extern crate libc;

use ffi::*;

use cuda::ffi::runtime::{cudaStream_t};
use float::stub::{f16_stub};
use libc::*;

use std::i32;
use std::ptr::{null_mut};

pub mod ffi;

pub trait NcclDataType {
  fn kind() -> ncclDataType_t;
}

impl NcclDataType for i8 {
  fn kind() -> ncclDataType_t { ncclDataType_t::Char }
}

impl NcclDataType for i32 {
  fn kind() -> ncclDataType_t { ncclDataType_t::Int }
}

impl NcclDataType for i64 {
  fn kind() -> ncclDataType_t { ncclDataType_t::Int64 }
}

impl NcclDataType for u64 {
  fn kind() -> ncclDataType_t { ncclDataType_t::Uint64 }
}

impl NcclDataType for f16_stub {
  fn kind() -> ncclDataType_t { ncclDataType_t::Half }
}

impl NcclDataType for f32 {
  fn kind() -> ncclDataType_t { ncclDataType_t::Float }
}

impl NcclDataType for f64 {
  fn kind() -> ncclDataType_t { ncclDataType_t::Double }
}

pub trait NcclOp {
  fn kind() -> ncclRedOp_t;
}

pub struct NcclSumOp;

impl NcclOp for NcclSumOp {
  fn kind() -> ncclRedOp_t { ncclRedOp_t::Sum }
}

pub struct NcclProdOp;

impl NcclOp for NcclProdOp {
  fn kind() -> ncclRedOp_t { ncclRedOp_t::Prod }
}

pub struct NcclMaxOp;

impl NcclOp for NcclMaxOp {
  fn kind() -> ncclRedOp_t { ncclRedOp_t::Max }
}

pub struct NcclMinOp;

impl NcclOp for NcclMinOp {
  fn kind() -> ncclRedOp_t { ncclRedOp_t::Min }
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

  pub unsafe fn reduce<T, Op>(&self, src: *const T, dst: *mut T, len: usize, _op: Op, root: usize, stream: cudaStream_t) -> Result<(), ncclResult_t>
  where T: NcclDataType, Op: NcclOp {
    assert!(len <= i32::MAX as usize);
    let res = ncclReduce(src as *const _, dst as *mut _, len as c_int, T::kind(), Op::kind(), root as c_int, self.ptr, stream);
    if res != ncclResult_t::Success {
      return Err(res);
    }
    Ok(())
  }

  pub unsafe fn allreduce<T, Op>(&self, src: *const T, dst: *mut T, len: usize, _op: Op, stream: cudaStream_t) -> Result<(), ncclResult_t>
  where T: NcclDataType, Op: NcclOp {
    assert!(len <= i32::MAX as usize);
    let res = ncclAllReduce(src as *const _, dst as *mut _, len as c_int, T::kind(), Op::kind(), self.ptr, stream);
    if res != ncclResult_t::Success {
      return Err(res);
    }
    Ok(())
  }

  pub unsafe fn reduce_scatter<T, Op>(&self, src: *const T, _src_len: usize, dst: *mut T, dst_len: usize, _op: Op, stream: cudaStream_t) -> Result<(), ncclResult_t>
  where T: NcclDataType, Op: NcclOp {
    let res = ncclReduceScatter(src as *const _, dst as *mut _, dst_len as c_int, T::kind(), Op::kind(), self.ptr, stream);
    if res != ncclResult_t::Success {
      return Err(res);
    }
    Ok(())
  }

  pub unsafe fn broadcast<T>(&self, buf: *mut T, len: usize, root: usize, stream: cudaStream_t) -> Result<(), ncclResult_t>
  where T: NcclDataType {
    assert!(len <= i32::MAX as usize);
    let res = ncclBcast(buf as *mut _, len as c_int, T::kind(), root as c_int, self.ptr, stream);
    if res != ncclResult_t::Success {
      return Err(res);
    }
    Ok(())
  }

  pub unsafe fn allgather<T>(&self, src: *const T, src_len: usize, dst: *mut T, _dst_len: usize, stream: cudaStream_t) -> Result<(), ncclResult_t>
  where T: NcclDataType {
    let res = ncclAllGather(src as *const _, src_len as c_int, T::kind(), dst as *mut _, self.ptr, stream);
    if res != ncclResult_t::Success {
      return Err(res);
    }
    Ok(())
  }
}
