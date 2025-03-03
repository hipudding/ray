import logging
from types import ModuleType
from typing import TYPE_CHECKING, List, Optional, Tuple

import ray
from ray.exceptions import RayChannelError
from ray.experimental.channel.communicator import Communicator, TorchTensorAllocator
from ray.experimental.util.types import ReduceOp

if TYPE_CHECKING:
    import torch
    import torch_npu

# Logger for this module. It should be configured at the entry point
# into the program using Ray. Ray provides a default configuration at
# entry/init points.
logger = logging.getLogger(__name__)


class _HcclGroup(Communicator):
    """
    Represents an actor's HCCL communicator. This is the default HCCL communicator
    to be used in Compiled Graph if a custom communicator is not provided.

    This class is not thread-safe.
    """

    def __init__(
        self,
        world_size: int,
        comm_id: int,
        rank: Optional[int],
        actor_handles: List["ray.actor.ActorHandle"],
        acl_stream: Optional[int],
        use_communication_streams: bool = False,
    ):
        self._world_size = world_size
        self._rank: Optional[int] = rank
        self.hccl: Optional[ModuleType] = None
        self._actor_handles = actor_handles
        self._use_communication_streams = use_communication_streams

        if rank is not None:
            assert "NPU" in ray.cluster_resources(), "HCCL actor has no NPUs assigned"
            assert acl_stream is not None, "HCCL actor must specify aclrtStream"

            expected_rank = self.get_rank(ray.get_runtime_context().current_actor)
            assert (
                rank == expected_rank
            ), f"HCCL actor's rank {rank} does not match expected rank {expected_rank}"

            from ray.util.hccl import hccl

            self.hccl = hccl

            self._comm = self.hccl.HCCLCommunicator(world_size, comm_id, rank)
        else:
            # Driver does not have a rank.
            self._comm = None

        self._acl_stream:  None
        self._send_stream:  None
        self._recv_stream:  None
        if acl_stream is not None:
            assert rank is not None, "HCCL actor has no rank assigned"

            self._acl_stream = acl_stream

            if use_communication_streams:
                import torch
                import torch_npu

                self._send_stream = torch.npu.Stream().npu_stream
                self._recv_stream = torch.npu.Stream().npu_stream
            else:
                self._send_stream = self._cuda_stream
                self._recv_stream = self._cuda_stream

        self._closed = False

    def initialize(self, rank: int) -> None:
        pass

    def get_actor_handles(self) -> List["ray.actor.ActorHandle"]:
        return self._actor_handles

    def get_rank(self, actor: ray.actor.ActorHandle) -> int:
        actor_ids = [a._ray_actor_id for a in self._actor_handles]
        try:
            rank = actor_ids.index(actor._ray_actor_id)
        except ValueError:
            raise ValueError("Actor is not in the HCCL group.")
        return rank

    def get_self_rank(self) -> Optional[int]:
        return self._rank

    def get_world_size(self) -> int:
        return self._world_size

    def send(self, buf: "torch.Tensor", peer_rank: int) -> None:
        if self._closed:
            raise RayChannelError("HCCL group has been destroyed.")

        if self._use_communication_streams:
            # We observed that if all recv/compute/send operations run on GPU,
            # since there is no synchronization, the CPU execution loop may be
            # far ahead of the GPU operations and lead to runtime failures.
            # To avoid that, we synchronize on the send stream.
            # TODO(rui): find a better approach
            torch.npu.Stream.wait_stream(self._send_stream)

        # TODO(swang): Handle send/recv async HCCL errors such as network
        # failures.
        self._comm.send(
            self.buf.data_ptr(),
            buf.numel(),
            self.hccl.get_hccl_tensor_dtype(buf),
            peer_rank,
            self._send_stream,
        )

    def recv(
        self,
        shape: Tuple[int],
        dtype: "torch.dtype",
        peer_rank: int,
        allocator=Optional[TorchTensorAllocator],
    ) -> "torch.Tensor":
        if self._closed:
            raise RayChannelError("HCCL group has been destroyed.")
        assert allocator is not None, "HCCL group requires a tensor allocator"
        buf = allocator(shape, dtype)

        if self._use_communication_streams:
            # We observed that if all recv/compute/send operations run on GPU,
            # since there is no synchronization, the CPU execution loop may be
            # far ahead of the GPU operations and lead to runtime failures.
            # To avoid that, we synchronize on the recv stream.
            # TODO(rui): find a better approach
            torch.npu.Stream.wait_stream(self._recv_stream)

            self._comm.recv(
                self.buf.data_ptr(),
                buf.numel(),
                self.hccl.get_hccl_tensor_dtype(buf),
                peer_rank,
                self._recv_stream,
            )
        else:
            self._comm.recv(
                self.buf.data_ptr(),
                buf.numel(),
                self.hccl.get_hccl_tensor_dtype(buf),
                peer_rank,
                self._recv_stream,
            )

            # Buffer values are undefined if HCCL ops are aborted. Therefore, we
            # need to synchronize here and check that the channel is still open to
            # ensure that the receive buffer is valid.
            # TODO(swang): Avoid CUDA synchronization.
            torch.npu.Stream.wait_stream(self._acl_stream)

        if self._closed:
            raise RayChannelError("HCCL group has been destroyed.")
        return buf

    def allreduce(
        self,
        send_buf: "torch.Tensor",
        recv_buf: "torch.Tensor",
        op: ReduceOp = ReduceOp.SUM,
    ):
        if self._closed:
            raise RayChannelError("HCCL group has been destroyed.")

        assert send_buf.dtype == recv_buf.dtype, (
            "Ray Compiled Graph derived the dtype of recv_buf from send_buf, "
            "so send_buf and recv_buf must have the same dtype. "
            "If you see this error, please file an issue at Ray repository."
        )
        self._comm.allReduce(
            self.send_buf.data_ptr(),
            self.recv_buf.data_ptr(),
            send_buf.numel(),
            self.hccl.get_hccl_tensor_dtype(send_buf),
            op.value,
            self._acl_stream,
        )

        # Buffer values are undefined if HCCL ops are aborted. Therefore, we
        # need to synchronize here and check that the channel is still open to
        # ensure that the receive buffer is valid.
        # TODO(swang): Avoid CUDA synchronization.
        # TODO(wxdeng): Use check_async_error.
        torch.npu.Stream.wait_stream(self._acl_stream)
        if self._closed:
            raise RayChannelError(
                "HCCL group has been destroyed during allreduce operation. "
                "There may be a dtype mismatch between input tensors from "
                "different ranks."
            )

    @property
    def recv_stream(self) -> Optional["cp.cuda.ExternalStream"]:
        return nullcontext()

    @property
    def send_stream(self) -> Optional["cp.cuda.ExternalStream"]:
        return nullcontext()

    def destroy(self) -> None:
        """
        Destroy the HCCL group.
        """
        if self._closed:
            return

        self._closed = True

        if self._comm is not None:
            logger.info(
                "Destructing HCCL group on actor: "
                f"{ray.get_runtime_context().current_actor}"
            )
            # Abort *after* setting the _closed flag. This ensures that HCCL
            # ops that were blocked on a remote peer will see that the _closed
            # flag is True when they exit from the abort.
            self._comm.destroy()

    def get_transport_name(self) -> str:
        return "hccl"
