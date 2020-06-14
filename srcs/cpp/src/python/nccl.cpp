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
nccl_controller::nccl_controller(const KungFu_NCCLScope scope) : scope_(scope)
{
}

void nccl_controller::InitOnce()
{
    if (_gpu_collective.get() == nullptr) {
        if (scope_ == KungFu_NCCL_LOCAL) {
            _gpu_collective.reset(new_local_gpu_collective(*_default_peer));
        } else {
            _gpu_collective.reset(new_global_gpu_collective(*_default_peer));
        }
    }
}

int nccl_controller::Reduce(const Workspace &w, KungFu_Op op, DoneCallback done)
{
    _gpu_collective->reduce(w.sendbuf, w.recvbuf, w.count, w.dtype);
    done();
    return 0;
}

int nccl_controller::Broadcast(const Workspace &w, DoneCallback done)
{
    _gpu_collective->broadcast(w.sendbuf, w.recvbuf, w.count, w.dtype);
    done();
    return 0;
}

int nccl_controller::AllReduce(const Workspace &w, KungFu_Op op,
                               DoneCallback done)
{
    _gpu_collective->all_reduce(w.sendbuf, w.recvbuf, w.count, w.dtype);
    done();
    return 0;
}

void NCCLHelper::EnsureController(const KungFu_NCCLScope scope)
{
    auto &ptr = controllers_[scope];
    if (ptr.get() == nullptr) { ptr.reset(new nccl_controller(scope)); }
    ptr->InitOnce();
}

nccl_controller *NCCLHelper::GetController(const KungFu_NCCLScope scope)
{
    return controllers_.at(scope).get();
}

NCCLScheduler *NCCLHelper::CreateScheduler(const std::string &name,
                                           const KungFu_NCCLScope scope)
{
    if (schedulers_.count(name) > 0) {
        throw std::runtime_error("duplicated NCCLScheduler creation: " + name);
    }
    schedulers_[name] = std::make_unique<NCCLScheduler>(name, scope);
    return GetScheduler(name);
}

NCCLScheduler *NCCLHelper::GetScheduler(const std::string &name) const
{
    return schedulers_.at(name).get();
}
}  // namespace kungfu
