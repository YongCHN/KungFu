#include <kungfu.h>
#include <kungfu/python/init.h>

void kungfu_python_init_nccl()
{
    kungfu::_global_nccl_controller.reset(new kungfu::nccl_controller(true));
    kungfu::_local_nccl_controller.reset(new kungfu::nccl_controller(false));
}

void kungfu_python_finialize_nccl()
{
    kungfu::_nccl_order_group.reset(nullptr);
    kungfu::_global_nccl_controller.reset(nullptr);
    kungfu::_local_nccl_controller.reset(nullptr);
}

namespace kungfu
{
std::unique_ptr<nccl_controller> _global_nccl_controller;
std::unique_ptr<nccl_controller> _local_nccl_controller;

nccl_controller::nccl_controller(bool global) : _global(global) {}

void nccl_controller::InitOnce()
{
    if (_gpu_collective.get() == nullptr) {
        if (_global) {
            _gpu_collective.reset(new_global_gpu_collective(*_default_peer));
        } else {
            _gpu_collective.reset(new_local_gpu_collective(*_default_peer));
        }
    }
}

int nccl_controller::ScheduledAllReduce(DoneCallback ready, const void *sendbuf,
                                        void *recvbuf, int count,
                                        KungFu_Datatype dtype, KungFu_Op op,
                                        const char *name, DoneCallback done)
{
    kungfu::_nccl_order_group->Start(name, [=, comm = _gpu_collective.get()]() {
        ready();
        comm->all_reduce(sendbuf, recvbuf, count, dtype);
        done();
    });
    return 0;
}

int nccl_controller::AllReduce(const void *sendbuf, void *recvbuf, int count,
                               KungFu_Datatype dtype, KungFu_Op op,
                               const char *name, DoneCallback done)
{
    _gpu_collective->all_reduce(sendbuf, recvbuf, count, dtype);
    done();
    return 0;
}
}  // namespace kungfu
