#pragma once
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <kungfu.h>
#include <kungfu/nccl/gpu_collective.hpp>

extern "C" {
extern void kungfu_python_init();
extern void kungfu_python_init_nccl();

extern void kungfu_python_finialize();
extern void kungfu_python_finialize_nccl();

extern int kungfu_get_cuda_index();

// helpers APIs to access kungfu without tensorflow operators
extern uint64_t kungfu_uid();
extern int kungfu_detached();
extern int kungfu_rank();        // get current rank
extern int kungfu_size();        // get current size
extern int kungfu_local_rank();  // get current local rank
extern int kungfu_local_size();  // get current local size
extern void kungfu_barrier();

extern int kungfu_propose_new_size(int new_size);

enum KungFu_NCCLScope {
    KungFu_NCCL_GLOBAL,
    KungFu_NCCL_LOCAL,
};

typedef enum KungFu_NCCLScope KungFu_NCCLScope;
}

extern std::unique_ptr<kungfu::Peer> _default_peer;

namespace kungfu
{
// order_group wraps order_group_t
class order_group
{
    order_group_t *og_;
    std::map<std::string, int> ranks_;

  public:
    using Task = DoneCallback;

    order_group(const std::vector<std::string> &names,
                const std::vector<int32_t> &order);

    ~order_group();

    void Start(const std::string &name, const Task &task);

    std::vector<int32_t> Wait();
};

// Workspace holds metadata for AllReduce, Reduce, Broadcast
struct Workspace {
    const void *sendbuf;
    void *recvbuf;
    const int count;
    const KungFu_Datatype dtype;
};

class nccl_controller
{
    KungFu_NCCLScope scope_;
    std::unique_ptr<gpu_collective> _gpu_collective;

  public:
    nccl_controller(const KungFu_NCCLScope scope);

    void InitOnce();

    int Reduce(const Workspace &w, KungFu_Op op, DoneCallback done);

    int Broadcast(const Workspace &w, DoneCallback done);

    int AllReduce(const Workspace &w, KungFu_Op op, DoneCallback done);
};

class NCCLScheduler
{
    const bool auto_order_;
    const std::string name_;
    const KungFu_NCCLScope scope_;

    int counter_;
    std::vector<int32_t> order_;

    std::unique_ptr<order_group> order_group_;

    void ResetOrder(int n);

  public:
    NCCLScheduler(const std::string &name, const KungFu_NCCLScope scope);

    void Reset(const std::vector<std::string> &names);

    void Start(const std::string &name, const order_group::Task &task);
};

// NCCLHelper is a singleton class that contains NCCL related global variables
class NCCLHelper
{
    std::map<KungFu_NCCLScope, std::unique_ptr<nccl_controller>> controllers_;
    std::map<std::string, std::unique_ptr<NCCLScheduler>> schedulers_;

  public:
    void EnsureController(const KungFu_NCCLScope scope);

    nccl_controller *GetController(const KungFu_NCCLScope scope);

    NCCLScheduler *CreateScheduler(const std::string &name,
                                   const KungFu_NCCLScope scope);
    NCCLScheduler *GetScheduler(const std::string &name) const;
};
}  // namespace kungfu

extern std::unique_ptr<kungfu::NCCLHelper> _default_nccl_helper;
