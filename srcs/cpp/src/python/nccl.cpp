#include <kungfu.h>
#include <kungfu/python/init.h>

std::unique_ptr<kungfu::NCCLHelper> _default_nccl_helper;

void kungfu_python_init_nccl()
{
    _default_nccl_helper.reset(new kungfu::NCCLHelper);
}

void kungfu_python_finialize_nccl() { _default_nccl_helper.reset(nullptr); }

namespace kungfu
{
NCCLController::NCCLController(const KungFu_NCCLScope scope) : scope_(scope) {}

void NCCLController::InitOnce()
{
    if (_gpu_collective.get() == nullptr) {
        if (scope_ == KungFu_NCCL_LOCAL) {
            _gpu_collective.reset(new_local_gpu_collective(*_default_peer));
        } else {
            _gpu_collective.reset(new_global_gpu_collective(*_default_peer));
        }
    }
}

int NCCLController::Reduce(const Workspace &w, KungFu_Op op, DoneCallback done)
{
    _gpu_collective->reduce(w.sendbuf, w.recvbuf, w.count, w.dtype);
    done();
    return 0;
}

int NCCLController::Broadcast(const Workspace &w, DoneCallback done)
{
    _gpu_collective->broadcast(w.sendbuf, w.recvbuf, w.count, w.dtype);
    done();
    return 0;
}

int NCCLController::AllReduce(const Workspace &w, KungFu_Op op,
                              DoneCallback done)
{
    _gpu_collective->all_reduce(w.sendbuf, w.recvbuf, w.count, w.dtype);
    done();
    return 0;
}

NCCLController *NCCLHelper::EnsureController(const KungFu_NCCLScope scope)
{
    std::lock_guard<std::mutex> _lk(mu_);
    auto &ptr = controllers_[scope];
    if (ptr.get() == nullptr) {
        ptr.reset(new NCCLController(scope));
        ptr->InitOnce();
    }
    return ptr.get();
}

NCCLScheduler *NCCLHelper::EnsureScheduler(const std::string &name,
                                           const KungFu_NCCLScope scope)
{
    std::lock_guard<std::mutex> _lk(mu_);
    auto &ptr = schedulers_[name];
    if (ptr.get() == nullptr) { ptr.reset(new NCCLScheduler(name, scope)); }
    return ptr.get();
}
}  // namespace kungfu
