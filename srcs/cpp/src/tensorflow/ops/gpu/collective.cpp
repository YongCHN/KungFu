#include <chrono>
#include <thread>

#include <kungfu/tensorflow/ops.h>
#include <kungfu/utils/trace.hpp>
#include <tensorflow/stream_executor/stream.h>

namespace tensorflow
{
void spin_wait(perftools::gputools::Event *event, int ms = 100)
{
    TRACE_SCOPE(__func__);
    while (event->PollForStatus() ==
           perftools::gputools::Event::Status::kPending) {
        std::this_thread::sleep_for(std::chrono::microseconds(ms));
    }
}

perftools::gputools::Event *create_init_ready_event(OpKernelContext *context)
{
    auto device_context = context->op_device_context();
    auto executor       = device_context->stream()->parent();
    auto ready_event    = new perftools::gputools::Event(executor);
    ready_event->Init();
    device_context->stream()->ThenRecordEvent(ready_event);
    return ready_event;
}

void wait_delete_ready_event(perftools::gputools::Event *ready_event)
{
    spin_wait(ready_event);
    delete ready_event;
}

REGISTER_KUNGFU_OP(ScheduledLocalNcclReduce)
    .Attr("T: {int32, int64, float16, float32, float64}")
    .Attr("input_tensor_name: string")
    .Attr("scheduler: string")
    .Attr("scope: string")
    .Input("input: T")
    .Output("output: T")
    .SetShapeFn(shape_inference::UnchangedShape);

class ScheduledLocalNcclReduce : public AsyncOpKernel
{
    std::string input_tensor_name_;
    kungfu::NCCLScheduler *scheduler_;
    kungfu::NCCLController *nccl_;

  public:
    explicit ScheduledLocalNcclReduce(OpKernelConstruction *context)
        : AsyncOpKernel(context)
    {
        OP_REQUIRES_OK(context, context->GetAttr("input_tensor_name",
                                                 &input_tensor_name_));
        OP_REQUIRES(
            context, input_tensor_name_.size() >= 0,
            errors::InvalidArgument("input_tensor_name must not be empty"));
        std::string scheduler;
        OP_REQUIRES_OK(context, context->GetAttr("scheduler", &scheduler));
        std::string scope_name;
        OP_REQUIRES_OK(context, context->GetAttr("scope", &scope_name));
        OP_REQUIRES(context, kungfu::_nccl_scopes.count(scope_name) > 0,
                    errors::InvalidArgument("invalid scope"));
        const auto scope = kungfu::_nccl_scopes.at(scope_name);
        scheduler_ = _default_nccl_helper->EnsureScheduler(scheduler, scope);
        nccl_      = _default_nccl_helper->EnsureController(scope);
    }

    void ComputeAsync(OpKernelContext *context, DoneCallback done) override
    {
        const Tensor &input = context->input(0);
        Tensor *output      = nullptr;
        OP_REQUIRES_OK_ASYNC(
            context, context->allocate_output(0, input.shape(), &output), done);
        const auto w     = make_workspace(input, output);
        auto ready_event = create_init_ready_event(context);
        scheduler_->Start(input_tensor_name_, [=] {
            wait_delete_ready_event(ready_event);
            nccl_->Reduce(w, KungFu_SUM, done);
        });
    }
};

REGISTER_KUNGFU_KERNEL_BUILDER(ScheduledLocalNcclReduce, DEVICE_GPU);

REGISTER_KUNGFU_OP(ScheduledLocalNcclBroadcast)
    .Attr("T: {int32, int64, float16, float32, float64}")
    .Attr("input_tensor_name: string")
    .Attr("scheduler: string")
    .Attr("scope: string")
    .Input("input: T")
    .Output("output: T")
    .SetShapeFn(shape_inference::UnchangedShape);

class ScheduledLocalNcclBroadcast : public AsyncOpKernel
{
    std::string input_tensor_name_;
    kungfu::NCCLScheduler *scheduler_;
    kungfu::NCCLController *nccl_;

  public:
    explicit ScheduledLocalNcclBroadcast(OpKernelConstruction *context)
        : AsyncOpKernel(context)
    {
        OP_REQUIRES_OK(context, context->GetAttr("input_tensor_name",
                                                 &input_tensor_name_));
        OP_REQUIRES(
            context, input_tensor_name_.size() >= 0,
            errors::InvalidArgument("input_tensor_name must not be empty"));
        std::string scheduler;
        OP_REQUIRES_OK(context, context->GetAttr("scheduler", &scheduler));
        std::string scope_name;
        OP_REQUIRES_OK(context, context->GetAttr("scope", &scope_name));
        OP_REQUIRES(context, kungfu::_nccl_scopes.count(scope_name) > 0,
                    errors::InvalidArgument("invalid scope"));
        const auto scope = kungfu::_nccl_scopes.at(scope_name);
        scheduler_ = _default_nccl_helper->EnsureScheduler(scheduler, scope);
        nccl_      = _default_nccl_helper->EnsureController(scope);
    }

    void ComputeAsync(OpKernelContext *context, DoneCallback done) override
    {
        const Tensor &input = context->input(0);
        Tensor *output      = nullptr;
        OP_REQUIRES_OK_ASYNC(
            context, context->allocate_output(0, input.shape(), &output), done);
        const auto w     = make_workspace(input, output);
        auto ready_event = create_init_ready_event(context);
        scheduler_->Start(input_tensor_name_, [=] {
            wait_delete_ready_event(ready_event);
            nccl_->Broadcast(w, done);
        });
    }
};

REGISTER_KUNGFU_KERNEL_BUILDER(ScheduledLocalNcclBroadcast, DEVICE_GPU);

REGISTER_KUNGFU_OP(ScheduledNcclAllReduce)
    .Attr("T: {int32, int64, float16, float32, float64}")
    .Attr("input_tensor_name: string")
    .Attr("scheduler: string")
    .Attr("scope: string")
    .Input("input: T")
    .Output("output: T")
    .SetShapeFn(shape_inference::UnchangedShape);

class ScheduledNcclAllReduce : public AsyncOpKernel
{
    std::string input_tensor_name_;
    kungfu::NCCLScheduler *scheduler_;
    kungfu::NCCLController *nccl_;

  public:
    explicit ScheduledNcclAllReduce(OpKernelConstruction *context)
        : AsyncOpKernel(context)
    {
        OP_REQUIRES_OK(context, context->GetAttr("input_tensor_name",
                                                 &input_tensor_name_));
        OP_REQUIRES(
            context, input_tensor_name_.size() >= 0,
            errors::InvalidArgument("input_tensor_name must not be empty"));
        std::string scheduler;
        OP_REQUIRES_OK(context, context->GetAttr("scheduler", &scheduler));
        std::string scope_name;
        OP_REQUIRES_OK(context, context->GetAttr("scope", &scope_name));
        OP_REQUIRES(context, kungfu::_nccl_scopes.count(scope_name) > 0,
                    errors::InvalidArgument("invalid scope"));
        const auto scope = kungfu::_nccl_scopes.at(scope_name);
        scheduler_ = _default_nccl_helper->EnsureScheduler(scheduler, scope);
        nccl_      = _default_nccl_helper->EnsureController(scope);
    }

    void ComputeAsync(OpKernelContext *context, DoneCallback done) override
    {
        const Tensor &input = context->input(0);
        Tensor *output      = nullptr;
        OP_REQUIRES_OK_ASYNC(
            context, context->allocate_output(0, input.shape(), &output), done);
        const auto w     = make_workspace(input, output);
        auto ready_event = create_init_ready_event(context);
        scheduler_->Start(input_tensor_name_, [=] {
            wait_delete_ready_event(ready_event);
            nccl_->AllReduce(w, KungFu_SUM, done);
        });
    }
};

REGISTER_KUNGFU_KERNEL_BUILDER(ScheduledNcclAllReduce, DEVICE_GPU);

REGISTER_KUNGFU_OP(LocalNcclReduce)
    .Attr("T: {int32, int64, float16, float32, float64}")
    .Attr("input_tensor_name: string")
    .Input("input: T")
    .Output("output: T")
    .SetShapeFn(shape_inference::UnchangedShape);

class LocalNcclReduce : public AsyncOpKernel
{
    std::string input_tensor_name_;
    kungfu::NCCLController *nccl_;

  public:
    explicit LocalNcclReduce(OpKernelConstruction *context)
        : AsyncOpKernel(context)
    {
        OP_REQUIRES_OK(context, context->GetAttr("input_tensor_name",
                                                 &input_tensor_name_));
        OP_REQUIRES(
            context, input_tensor_name_.size() >= 0,
            errors::InvalidArgument("input_tensor_name must not be empty"));
        nccl_ = _default_nccl_helper->EnsureController(KungFu_NCCL_LOCAL);
    }

    void ComputeAsync(OpKernelContext *context, DoneCallback done) override
    {
        const Tensor &input = context->input(0);
        Tensor *output      = nullptr;
        OP_REQUIRES_OK_ASYNC(
            context, context->allocate_output(0, input.shape(), &output), done);
        wait_delete_ready_event(create_init_ready_event(context));
        nccl_->Reduce(make_workspace(input, output), KungFu_SUM, done);
    }
};

REGISTER_KUNGFU_KERNEL_BUILDER(LocalNcclReduce, DEVICE_GPU);

REGISTER_KUNGFU_OP(LocalNcclBroadcast)
    .Attr("T: {int32, int64, float16, float32, float64}")
    .Attr("input_tensor_name: string")
    .Input("input: T")
    .Output("output: T")
    .SetShapeFn(shape_inference::UnchangedShape);

class LocalNcclBroadcast : public AsyncOpKernel
{
    std::string input_tensor_name_;
    kungfu::NCCLController *nccl_;

  public:
    explicit LocalNcclBroadcast(OpKernelConstruction *context)
        : AsyncOpKernel(context)
    {
        OP_REQUIRES_OK(context, context->GetAttr("input_tensor_name",
                                                 &input_tensor_name_));
        OP_REQUIRES(
            context, input_tensor_name_.size() >= 0,
            errors::InvalidArgument("input_tensor_name must not be empty"));
        nccl_ = _default_nccl_helper->EnsureController(KungFu_NCCL_LOCAL);
    }

    void ComputeAsync(OpKernelContext *context, DoneCallback done) override
    {
        const Tensor &input = context->input(0);
        Tensor *output      = nullptr;
        OP_REQUIRES_OK_ASYNC(
            context, context->allocate_output(0, input.shape(), &output), done);
        wait_delete_ready_event(create_init_ready_event(context));
        nccl_->Broadcast(make_workspace(input, output), done);
    }
};

REGISTER_KUNGFU_KERNEL_BUILDER(LocalNcclBroadcast, DEVICE_GPU);

REGISTER_KUNGFU_OP(NcclAllReduce)
    .Attr("T: {int32, int64, float16, float32, float64}")
    .Attr("input_tensor_name: string")
    .Input("input: T")
    .Output("output: T")
    .SetShapeFn(shape_inference::UnchangedShape);

class NcclAllReduce : public AsyncOpKernel
{
    std::string input_tensor_name_;
    kungfu::NCCLController *nccl_;

  public:
    explicit NcclAllReduce(OpKernelConstruction *context)
        : AsyncOpKernel(context)
    {
        OP_REQUIRES_OK(context, context->GetAttr("input_tensor_name",
                                                 &input_tensor_name_));
        OP_REQUIRES(
            context, input_tensor_name_.size() >= 0,
            errors::InvalidArgument("input_tensor_name must not be empty"));
        nccl_ = _default_nccl_helper->EnsureController(KungFu_NCCL_GLOBAL);
    }

    void ComputeAsync(OpKernelContext *context, DoneCallback done) override
    {
        const Tensor &input = context->input(0);
        Tensor *output      = nullptr;
        OP_REQUIRES_OK_ASYNC(
            context, context->allocate_output(0, input.shape(), &output), done);
        wait_delete_ready_event(create_init_ready_event(context));
        nccl_->AllReduce(make_workspace(input, output), KungFu_SUM, done);
    }
};

REGISTER_KUNGFU_KERNEL_BUILDER(NcclAllReduce, DEVICE_GPU);
}  // namespace tensorflow
