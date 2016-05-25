extern crate gcc;

fn main() {
  gcc::Config::new()
    .compiler("/usr/local/cuda/bin/nvcc")
    .opt_level(3)
    // FIXME(20151207): for working w/ K80.
    .flag("-arch=sm_37")
    //.flag("-arch=sm_50")
    .flag("-maxrregcount=96")
    .flag("-ccbin=g++-4.8")
    .flag("-std=c++11")
    //.flag("-Xcompiler='-std=gnu++0x'")
    .pic(true)
    .include("../nccl/src")
    .include("/usr/local/cuda/include")
    .file("../nccl/src/all_gather.cu")
    .file("../nccl/src/all_reduce.cu")
    .file("../nccl/src/broadcast.cu")
    .file("../nccl/src/core.cu")
    .file("../nccl/src/libwrap.cu")
    .file("../nccl/src/reduce.cu")
    .file("../nccl/src/reduce_scatter.cu")
    .compile("libnccl_native.a");
}
