import os
import sys
import torch
import torch_npu

import ray
import ray.cluster_utils
from ray.dag import InputNode
from ray.air._internal import torch_utils

import pdb


@ray.remote(resources={"NPU": 1})
class NPUTorchTensorWorker:
    def __init__(self):
        self.device = torch_utils.get_devices()[0]
        print(self.device)
        from ray.air._internal.device_manager import get_torch_device_manager_by_context
        get_torch_device_manager_by_context().set_device(self.device)

    def send(self, shape, dtype, value: int, send_tensor=True):
        if not send_tensor:
            return 1
        return torch.ones(shape, device=self.device, dtype=dtype) * value

    def recv(self, tensor):
        assert tensor.device == self.device
        return (tensor[0].item(), tensor.shape, tensor.dtype)

    def return_tensor(self, size: int) -> torch.Tensor:
        return torch.ones(size)



def test_p2p_basic():
    sender = NPUTorchTensorWorker.remote()
    receiver = NPUTorchTensorWorker.remote()

    shape = (10,)
    dtype = torch.float16

    with InputNode() as inp:
        dag = sender.send.bind(inp.shape, inp.dtype, inp[0])
        dag = dag.with_tensor_transport(transport='hccl')
        dag = receiver.recv.bind(dag)

    compiled_dag = dag.experimental_compile()
    ref = compiled_dag.execute(5, shape=shape, dtype=dtype)
    assert ray.get(ref) == (5, shape, dtype)



if __name__ == "__main__":
    ray.init(num_cpus=2, resources={"NPU": 2})
    test_p2p_basic()