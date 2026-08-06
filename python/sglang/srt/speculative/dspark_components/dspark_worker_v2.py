import contextlib
import logging

import torch

from sglang.srt.layers.moe.utils import (
    speculative_moe_a2a_backend_context,
    speculative_moe_backend_context,
)
from sglang.srt.managers.schedule_batch import ModelWorkerBatch
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
    compute_position,
)
from sglang.srt.model_executor.cuda_graph_runner import CudaGraphRunner
from sglang.srt.speculative.base_spec_worker import BaseDraftWorker
from sglang.srt.speculative.eagle_info import EagleDraftInput, EagleVerifyInput
from sglang.srt.speculative.eagle_utils import (
    TreeMaskMode,
    build_tree_kernel_efficient,
)
from sglang.srt.speculative.eagle_worker_v2 import EAGLEWorkerV2, _get_plan_stream
from sglang.srt.utils.common import empty_context

logger = logging.getLogger(__name__)


class DSparkDraftWorker(BaseDraftWorker):
    def __init__(
        self,
        server_args,
        gpu_id,
        tp_rank,
        dp_rank,
        moe_ep_rank,
        attn_cp_rank,
        moe_dp_rank,
        nccl_port,
        target_worker,
    ):
        self.server_args = server_args
        self.target_worker = target_worker
        self.device = server_args.device
        self.gamma = int(server_args.speculative_num_steps)
        self.verify_tokens = self.gamma + 1
        self.req_to_token_pool, allocator = target_worker.get_memory_pool()

        old_disable_graph = server_args.disable_cuda_graph
        old_disable_piecewise_graph = server_args.disable_piecewise_cuda_graph
        server_args.disable_cuda_graph = True
        server_args.disable_piecewise_cuda_graph = True
        with speculative_moe_backend_context(), speculative_moe_a2a_backend_context():
            self.draft_worker = TpModelWorker(
                server_args=server_args,
                gpu_id=gpu_id,
                tp_rank=tp_rank,
                pp_rank=0,
                dp_rank=dp_rank,
                moe_ep_rank=moe_ep_rank,
                attn_cp_rank=attn_cp_rank,
                moe_dp_rank=moe_dp_rank,
                nccl_port=nccl_port,
                is_draft_worker=True,
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool_allocator=allocator,
                memory_pool_config=target_worker.model_runner.memory_pool_config,
            )
        server_args.disable_cuda_graph = old_disable_graph
        server_args.disable_piecewise_cuda_graph = old_disable_piecewise_graph
        self.draft_runner = self.draft_worker.model_runner
        self.model = self.draft_runner.model
        self.config = self.model.dspark_config
        self.model.set_runtime_gamma(self.gamma)
        self.model.attach_target_modules(target_worker.model_runner.model)
        target_worker.model_runner.model.set_eagle3_layers_to_capture(
            list(self.config.target_layer_ids)
        )
        self.draft_tp_context = empty_context
        self._tree_inputs = {}
        self.draft_cuda_graph_runner = None
        if (
            not old_disable_graph
            and torch.cuda.is_available()
        ):
            # DSpark's draft graph has gamma tokens per request.  This differs
            # from the target verify graph, which has gamma + one bonus token.
            # Capture the parallel draft block as fixed-width EXTEND so the
            # ENCODER_ONLY draft attention remains bidirectional.
            self.model._capture_greedy_proposal = True
            try:
                self.draft_cuda_graph_runner = CudaGraphRunner(
                    self.draft_runner,
                    capture_forward_mode_override=ForwardMode.EXTEND,
                    num_tokens_per_bs_override=self.gamma,
                    capture_hidden_mode_override=CaptureHiddenMode.FULL,
                    seq_len_fill_value_override=self.gamma,
                    spec_info_factory=lambda _: None,
                )
            finally:
                self.model._capture_greedy_proposal = False
            if tp_rank == 0:
                logger.info(
                    "Captured DSpark static draft CUDA graphs (block_size=%d, "
                    "max_bs=%d) with folded greedy proposal.",
                    self.gamma,
                    self.draft_cuda_graph_runner.max_bs,
                )

    def draft_extend(self):
        pass

    def draft(self, batch):
        return self.propose(batch)

    def _inject(self, hidden, positions, cache_loc):
        if hidden is None or hidden.numel() == 0:
            return
        self.model.write_target_hidden_kv(
            hidden,
            positions.to(dtype=torch.long),
            cache_loc.to(dtype=torch.long),
            self.draft_runner.token_to_kv_pool,
        )

    def inject_prefill(self, batch, hidden):
        extend_lens = torch.tensor(
            batch.extend_seq_lens, dtype=torch.int32, device=self.device
        )
        prefix_lens = torch.tensor(
            batch.extend_prefix_lens, dtype=torch.int32, device=self.device
        )
        positions, _ = compute_position(
            self.server_args.attention_backend,
            prefix_lens,
            extend_lens,
            int(batch.extend_num_tokens),
        )
        self._inject(hidden, positions, batch.out_cache_loc)

    @contextlib.contextmanager
    def _draft_batch(self, batch, input_ids, positions, cache_loc):
        fields = (
            "forward_mode",
            "input_ids",
            "out_cache_loc",
            "seq_lens",
            "seq_lens_cpu",
            "seq_lens_sum",
            "extend_seq_lens",
            "extend_prefix_lens",
            "extend_num_tokens",
            "capture_hidden_mode",
            "spec_info",
        )
        saved = {name: getattr(batch, name) for name in fields}
        bs = len(saved["seq_lens"])
        prefix_cpu = saved["seq_lens_cpu"]
        if prefix_cpu is None:
            prefix_cpu = saved["seq_lens"].cpu()
        batch.forward_mode = ForwardMode.EXTEND
        batch.input_ids = input_ids
        batch.out_cache_loc = cache_loc
        batch.seq_lens = saved["seq_lens"] + self.gamma
        batch.seq_lens_cpu = prefix_cpu + self.gamma
        batch.seq_lens_sum = int(batch.seq_lens_cpu.sum())
        batch.extend_seq_lens = [self.gamma] * bs
        batch.extend_prefix_lens = prefix_cpu.tolist()
        batch.extend_num_tokens = bs * self.gamma
        batch.capture_hidden_mode = CaptureHiddenMode.FULL
        batch.spec_info = EagleDraftInput(
            num_tokens_per_req=self.gamma,
            num_tokens_for_logprob_per_req=self.gamma,
        )
        batch.spec_info.positions = positions
        try:
            yield ForwardBatch.init_new(batch, self.draft_runner)
        finally:
            for name, value in saved.items():
                setattr(batch, name, value)

    def propose(self, batch: ModelWorkerBatch) -> EagleVerifyInput:
        draft_input = batch.spec_info
        if batch.forward_mode.is_idle():
            return EagleVerifyInput.create_idle_input(
                1, self.gamma, self.verify_tokens
            )
        if not isinstance(draft_input, EagleDraftInput):
            raise TypeError(
                f"DSpark expected EagleDraftInput state, got {type(draft_input)}."
            )

        bs = len(batch.seq_lens)
        offsets = torch.arange(self.gamma, device=self.device)
        positions_2d = batch.seq_lens[:, None] + offsets[None, :]
        cache_loc = self.req_to_token_pool.req_to_token[
            batch.req_pool_indices[:, None], positions_2d
        ]
        block_ids = torch.full(
            (bs, self.gamma),
            self.config.mask_token_id,
            dtype=torch.long,
            device=self.device,
        )
        block_ids[:, 0] = draft_input.verified_id.view(-1)

        with self._draft_batch(
            batch,
            block_ids.flatten(),
            positions_2d.flatten(),
            cache_loc.flatten(),
        ) as forward_batch:
            can_run_draft_graph = (
                batch.sampling_info.is_all_greedy
                and self.draft_cuda_graph_runner is not None
                and self.draft_cuda_graph_runner.can_run(forward_batch)
            )
            if can_run_draft_graph:
                output = self.draft_cuda_graph_runner.replay(forward_batch)
            else:
                output = self.draft_runner.forward(forward_batch).logits_output
        anchors = draft_input.verified_id.view(-1)
        if can_run_draft_graph:
            draft_tokens = output.customized_info["dspark_draft_tokens"][:bs]
        else:
            draft_tokens = self.model.sample_block(
                output.hidden_states,
                anchors,
                batch.sampling_info,
            )

        tree_mask_buf, position_buf = (
            self.target_worker.model_runner.attn_backend.get_verify_buffers_to_fill_after_draft()
        )
        tree_inputs = self._tree_inputs.get(bs)
        if tree_inputs is None:
            tree_inputs = (
                torch.arange(
                    -1, self.gamma - 1, dtype=torch.long, device=self.device
                ).repeat(bs, 1),
                torch.arange(
                    self.gamma, dtype=torch.long, device=self.device
                ).repeat(bs, 1),
            )
            self._tree_inputs[bs] = tree_inputs
        parents, selected = tree_inputs
        (
            tree_mask,
            positions,
            retrieve_index,
            retrieve_next_token,
            retrieve_next_sibling,
            candidates,
        ) = build_tree_kernel_efficient(
            draft_input.verified_id.view(-1),
            parents,
            selected,
            draft_tokens,
            batch.seq_lens,
            batch.seq_lens_sum,
            1,
            self.gamma,
            self.verify_tokens,
            TreeMaskMode.FULL_MASK,
            tree_mask_buf,
            position_buf,
        )
        return EagleVerifyInput(
            draft_token=candidates,
            custom_mask=tree_mask,
            positions=positions,
            retrive_index=retrieve_index,
            retrive_next_token=retrieve_next_token,
            retrive_next_sibling=retrieve_next_sibling,
            retrive_cum_len=None,
            spec_steps=self.gamma,
            topk=1,
            draft_token_num=self.verify_tokens,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            seq_lens_sum=None,
            seq_lens_cpu=None,
        )


class DSparkWorkerV2(EAGLEWorkerV2):
    """Static DSpark proposal with the existing EAGLE V2 verify/accept path."""

    def __init__(
        self,
        server_args,
        gpu_id,
        tp_rank,
        dp_rank,
        moe_ep_rank,
        attn_cp_rank,
        moe_dp_rank,
        nccl_port,
        target_worker,
    ):
        self.server_args = server_args
        self.topk = 1
        self.speculative_num_steps = int(server_args.speculative_num_steps)
        self.speculative_num_draft_tokens = self.speculative_num_steps + 1
        self.tp_rank = tp_rank
        self.gpu_id = gpu_id
        self.device = server_args.device
        self._target_worker = target_worker
        self.page_size = server_args.page_size
        self.req_to_token_pool, self.token_to_kv_pool_allocator = (
            target_worker.get_memory_pool()
        )
        server_args.context_length = target_worker.model_runner.model_config.context_len
        self._draft_worker = DSparkDraftWorker(
            server_args,
            gpu_id,
            tp_rank,
            dp_rank,
            moe_ep_rank,
            attn_cp_rank,
            moe_dp_rank,
            nccl_port,
            target_worker,
        )
        self.plan_stream, self.plan_stream_ctx = _get_plan_stream(self.device)

    @staticmethod
    def _fill_overlap_fields(draft_input: EagleDraftInput):
        """Populate the EAGLE-shaped fields used by the overlap future map.

        DSpark generates its whole block in ``propose`` and therefore does not
        consume EAGLE's top-k probabilities.  The scheduler still stores these
        fields for every spec-v2 worker, so keep shape-compatible placeholders.
        """
        verified_id = draft_input.verified_id.view(-1)
        draft_input.topk_p = torch.ones(
            (verified_id.numel(), 1),
            dtype=torch.float32,
            device=verified_id.device,
        )
        draft_input.topk_index = verified_id.to(dtype=torch.long).view(-1, 1)

    def forward_batch_generation(self, batch: ModelWorkerBatch):
        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            batch.capture_hidden_mode = CaptureHiddenMode.FULL
            output = self.target_worker.forward_batch_generation(batch)
            self.draft_worker.inject_prefill(
                batch, output.logits_output.hidden_states
            )
            output.logits_output.hidden_states = None
            output.next_draft_input = EagleDraftInput(
                verified_id=output.next_token_ids,
                new_seq_lens=batch.seq_lens,
            )
            self._fill_overlap_fields(output.next_draft_input)
            return output

        if batch.spec_info is None:
            batch.spec_info = EagleDraftInput.create_idle_input(
                device=self.device,
                hidden_size=self.target_worker.model_config.hidden_size,
                dtype=self.target_worker.model_config.dtype,
                topk=1,
                capture_hidden_mode=CaptureHiddenMode.LAST,
            )
        verify_input = self.draft_worker.propose(batch)
        batch.spec_info = verify_input
        output = self.verify(batch)
        if not batch.forward_mode.is_idle():
            self.draft_worker._inject(
                output.logits_output.hidden_states,
                verify_input.positions,
                batch.out_cache_loc,
            )
            output.logits_output.hidden_states = None
        self._fill_overlap_fields(output.next_draft_input)
        return output
