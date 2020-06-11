#include <kungfu/tensorflow/ops.h>

namespace tensorflow
{
static const std::set<std::string> kungfu_nccl_scopes({
    // "local", // TODO: support local
    "global",
});

REGISTER_KUNGFU_OP(StartNcclScheduler)
    .Attr("scope: string")  // local | global
    .Input("input: string");

class StartNcclScheduler : public OpKernel
{
    int counter_;
    std::vector<int32_t> order_;

    void ResetOrder(int n)
    {
        order_.resize(n);
        std::iota(order_.begin(), order_.end(), 0);
    }

  public:
    explicit StartNcclScheduler(OpKernelConstruction *context)
        : OpKernel(context), counter_(0)
    {
        std::string scope;
        OP_REQUIRES_OK(context, context->GetAttr("scope", &scope));
        OP_REQUIRES(context, kungfu_nccl_scopes.count(scope) > 0,
                    errors::InvalidArgument("invalid scope"));
        kungfu::_global_nccl_controller->InitOnce();
    }

    void Compute(OpKernelContext *context) override
    {
        const Tensor &input = context->input(0);
        const auto t_names  = input.vec<std::string>();
        std::vector<std::string> names;
        for (int i = 0; i < t_names.size(); ++i) {
            names.push_back(t_names(i));
        }
        if (names.size() != order_.size()) { ResetOrder(names.size()); }
        if (kungfu::_nccl_order_group.get() != nullptr) {
            if (counter_ == 1) {
                const std::vector<int32_t> arrive_order =
                    kungfu::_nccl_order_group->Wait();
                if (arrive_order.size() == order_.size()) {
                    _default_peer->Broadcast(
                        arrive_order.data(), order_.data(), order_.size(),
                        to_kungfu_type(DT_INT32), name().c_str());
                }
            }
        }
        kungfu::_nccl_order_group.reset(new kungfu::order_group(names, order_));
        ++counter_;
    }
};

REGISTER_KUNGFU_KERNEL_BUILDER(StartNcclScheduler, DEVICE_CPU);
}  // namespace tensorflow
