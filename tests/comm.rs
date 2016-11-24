//extern crate devicemem_cuda;
extern crate nccl;

use nccl::*;

#[test]
fn comm_test() {
  let comm_id = NcclUniqueId::create().unwrap();
  let comm = NcclComm::create(0, 1, comm_id).unwrap();
}
