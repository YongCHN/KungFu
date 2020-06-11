#include <kungfu.h>
#include <kungfu/python/init.h>

std::unique_ptr<kungfu::TFNCCLHelper> _default_nccl_helper;

void kungfu_python_init_nccl()
{
    _default_nccl_helper.reset(new kungfu::TFNCCLHelper);
}

void kungfu_python_finialize_nccl() { _default_nccl_helper.reset(nullptr); }

namespace kungfu
{
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
    _default_nccl_helper->_nccl_order_group->Start(
        name, [=, comm = _gpu_collective.get()]() {
            ready();
            comm->all_reduce(sendbuf, recvbuf, count, dtype);
            done();
        });
    return 0;
}

int nccl_controller::Reduce(const void *sendbuf, void *recvbuf, int count,
                            KungFu_Datatype dtype, KungFu_Op op,
                            const char *name, DoneCallback done)
{
    _gpu_collective->reduce(sendbuf, recvbuf, count, dtype);
    done();
    return 0;
}

int nccl_controller::Broadcast(const void *sendbuf, void *recvbuf, int count,
                               KungFu_Datatype dtype, const char *name,
                               DoneCallback done)
{
    _gpu_collective->broadcast(sendbuf, recvbuf, count, dtype);
    done();
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

TFNCCLHelper::TFNCCLHelper()
{
    _global_nccl_controller.reset(new nccl_controller(true));
    _local_nccl_controller.reset(new nccl_controller(false));
}
}  // namespace kungfu
