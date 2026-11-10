"""
Supplementary SkeletonGPT module for SPRig.

This module post-trains the skeleton component of Seed3D/Puppeteer with
weighted cross-pose token distillation. Hard-decoded geometry is computed
without gradients and is used only to select a checkpoint on validation data.
"""

import torch
from torch import nn
from transformers import AutoModelForCausalLM

from skeleton_models.skeleton_opt_finetune import SkeletonOPTConfig
from third_partys.Michelangelo.encode import load_model


def undiscretize(t, low, high, num_discrete):
    assert (t >= 0).all() and (t <= num_discrete - 1).all()
    assert high > low
    t = t.float()
    t /= num_discrete
    t = t * (high - low) + low
    return t


class SkeletonGPT(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.point_encoder = load_model()
        self.cond_length = 257
        self.cond_dim = 768
        self.joint_token = True
        self.joint_per_token = 4
        self.bone_per_token = self.joint_per_token
        self.n_discrete_size = args.n_discrete_size
        effective_n_max_bones = args.n_max_bones + 1 if self.joint_token else args.n_max_bones
        self.max_length = int(effective_n_max_bones * self.bone_per_token + 2 + self.cond_length)
        self.pad_id = -1
        self.coor_continuous_range = (-0.5, 0.5)
        vocab_size = self.n_discrete_size + 3

        self.config = SkeletonOPTConfig.from_pretrained(
            args.llm,
            n_positions=self.max_length,
            max_position_embeddings=self.max_length,
            vocab_size=vocab_size,
            _attn_implementation="flash_attention_2",
        )

        self.bos_token_id = 0
        self.eos_token_id = 1
        self.pad_token_id = 2

        self.config.joint_token = self.joint_token
        self.config.bos_token_id = self.bos_token_id
        self.config.eos_token_id = self.eos_token_id
        self.config.pad_token_id = self.pad_token_id
        self.config.n_discrete_size = self.n_discrete_size
        self.config.bone_per_token = self.bone_per_token
        self.config.cond_length = self.cond_length
        self.config.word_embed_proj_dim = self.config.hidden_size

        if self.args.seq_shuffle:
            self.feat_dim = self.config.word_embed_proj_dim
            self.target_aware_pos_embed = nn.Parameter(
                torch.zeros(1, args.n_max_bones + 1, self.config.word_embed_proj_dim)
            )
            nn.init.trunc_normal_(self.target_aware_pos_embed, 0.0, 0.02)

        self.transformer = AutoModelForCausalLM.from_config(
            config=self.config,
            attn_implementation="flash_attention_2",
        )
        self.cond_head_proj = nn.Linear(self.cond_dim, self.config.word_embed_proj_dim)
        self.cond_proj = nn.Linear(self.cond_dim, self.config.word_embed_proj_dim)

    def detokenize_joint_token(self, input_ids):
        batch_size = input_ids.size(0)
        bones_coors_list = []
        num_bones_list = []

        for i in range(batch_size):
            cur_ids = input_ids[i]
            cur_ids = cur_ids[cur_ids != self.pad_id]

            if cur_ids.numel() % 4 != 0:
                bones_coors_list.append(
                    torch.empty(0, 2, 3, device=input_ids.device, dtype=torch.float32)
                )
                num_bones_list.append(0)
                continue

            num_joints = cur_ids.numel() // 4
            if num_joints == 0:
                bones_coors_list.append(
                    torch.empty(0, 2, 3, device=input_ids.device, dtype=torch.float32)
                )
                num_bones_list.append(0)
                continue

            joint_data = cur_ids.view(num_joints, 4)
            coords_discrete = joint_data[:, :3]
            coords_float = undiscretize(
                coords_discrete,
                self.coor_continuous_range[0],
                self.coor_continuous_range[1],
                self.n_discrete_size,
            )
            parents = joint_data[:, 3]

            bone_coords = []
            for child_idx in range(num_joints):
                p = parents[child_idx].item()
                if p > 0:
                    try:
                        parent_idx = p - 1
                        if parent_idx >= num_joints or parent_idx < 0:
                            continue
                        parent_coord = coords_float[parent_idx]
                        child_coord = coords_float[child_idx]
                        bone_coords.append(torch.stack([parent_coord, child_coord]))
                    except IndexError:
                        continue

            if not bone_coords:
                stacked_bone_coords = torch.empty(
                    0,
                    2,
                    3,
                    device=input_ids.device,
                    dtype=torch.float32,
                )
            else:
                stacked_bone_coords = torch.stack(bone_coords, dim=0)

            bones_coors_list.append(stacked_bone_coords)
            num_bones_list.append(stacked_bone_coords.size(0))

        if not bones_coors_list:
            return torch.full(
                (batch_size, 1, 2, 3),
                float('nan'),
                dtype=torch.float32,
                device=input_ids.device,
            )

        max_num_bones = max(num_bones_list) if num_bones_list else 0
        if max_num_bones == 0:
            max_num_bones = 1

        continuous_coors = torch.full(
            (batch_size, max_num_bones, 2, 3),
            float('nan'),
            dtype=torch.float32,
            device=input_ids.device,
        )
        for i in range(batch_size):
            num_bones = num_bones_list[i]
            if num_bones > 0:
                continuous_coors[i, :num_bones, :, :] = bones_coors_list[i]
        return continuous_coors

    def process_point_feature(self, point_feature):
        current_batch_size = point_feature.shape[0]
        encode_feature = torch.zeros(
            current_batch_size,
            self.cond_length,
            self.config.word_embed_proj_dim,
            device=self.cond_head_proj.weight.device,
            dtype=self.cond_head_proj.weight.dtype,
        )
        encode_feature[:, 0] = self.cond_head_proj(point_feature[:, 0])
        shape_latents = self.point_encoder.to_shape_latents(point_feature[:, 1:])
        encode_feature[:, 1:] = self.cond_proj(shape_latents)
        return encode_feature

    def generate(self, pc_normal, deterministic: bool = False):
        """
        Generate skeleton edges from a 6D point cloud.

        When `deterministic` is True, sampling is disabled and beam search is
        used for evaluation or validation. The original training-time
        generation settings are restored before returning.
        """
        was_training = self.training
        old_do_sample = getattr(self.args, "do_sample", False)
        old_num_beams = getattr(self.args, "num_beams", 1)

        try:
            self.eval()

            if deterministic:
                self.args.do_sample = False
                self.args.num_beams = max(1, getattr(self.args, "num_beams_eval", 4))
                self.args.top_p = None
                self.args.top_k = None

            point_feature = self.point_encoder.encode_latents(pc_normal)
            processed_point_feature = self.process_point_feature(point_feature)
            generate_length = self.max_length - self.cond_length

            if self.args.seq_shuffle:
                num_joint_token = self.max_length - 2 - self.cond_length
                num_joints = num_joint_token // self.joint_per_token
                if not hasattr(self, 'target_aware_pos_embed'):
                    raise AttributeError(
                        "Model is missing 'target_aware_pos_embed'. Initialize it in __init__ when seq_shuffle is True."
                    )
                target_aware_pos_embed = self.target_aware_pos_embed.repeat(pc_normal.shape[0], 1, 1)
                cond_pos_embed = target_aware_pos_embed[:, 0:1, :].repeat(1, self.cond_length, 1)
                bone_pos_embed = target_aware_pos_embed[:, 1:num_joints, :]
                bone_pos_embed_expanded = bone_pos_embed.unsqueeze(2).repeat(
                    1,
                    1,
                    self.joint_per_token,
                    1,
                )
                bone_pos_embed_expanded = bone_pos_embed_expanded.view(
                    pc_normal.shape[0],
                    num_joint_token - self.joint_per_token,
                    self.config.word_embed_proj_dim,
                )
                processed_point_feature += cond_pos_embed
            else:
                bone_pos_embed_expanded = None

            # Some HuggingFace versions do not expose `target_pos_embed` in
            # generate(); in that case we silently fall back to the standard call.
            generate_kwargs = {
                "inputs_embeds": processed_point_feature,
                "max_new_tokens": generate_length,
                "num_beams": getattr(self.args, "num_beams", 1),
                "do_sample": getattr(self.args, "do_sample", False),
                "top_k": getattr(self.args, "top_k", 50),
                "top_p": getattr(self.args, "top_p", 0.95),
                "bos_token_id": self.bos_token_id,
                "eos_token_id": self.eos_token_id,
                "pad_token_id": self.pad_token_id,
            }
            if (
                bone_pos_embed_expanded is not None
                and 'target_pos_embed' in self.transformer.generate.__code__.co_varnames
            ):
                generate_kwargs["target_pos_embed"] = bone_pos_embed_expanded

            amp_enabled = torch.cuda.is_available()
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                results_tokens = self.transformer.generate(**generate_kwargs)

            outputs = results_tokens[:, 1:-1]
            outputs[outputs == self.bos_token_id] = self.pad_id
            outputs[outputs == self.eos_token_id] = self.pad_id
            outputs[outputs == self.pad_token_id] = self.pad_id
            outputs[outputs != self.pad_id] -= 3

            if self.joint_token:
                pred_bone_coords = self.detokenize_joint_token(outputs)
            else:
                raise NotImplementedError("detokenize for non-joint-token is not implemented.")

            return pred_bone_coords
        finally:
            self.args.do_sample = old_do_sample
            self.args.num_beams = old_num_beams
            if was_training:
                self.train()

    @torch.no_grad()
    def decoded_geometry_score_against_anchor(
        self,
        keyframe_meshes: torch.Tensor,
        anchor_labels: torch.Tensor,
    ):
        """
        Score freely decoded non-anchor skeletons against the anchor skeleton.

        This validation-only score is outside the gradient graph. It combines:
        1. dominant edge-direction agreement,
        2. edge-length distribution agreement, and
        3. endpoint-set Chamfer agreement after rigid alignment.
        """
        dir_w = getattr(self.args, "lambda_dir", 1.0)
        len_w = getattr(self.args, "lambda_len", 0.2)
        chamfer_w = getattr(self.args, "lambda_chamfer", 0.5)
        topk_ratio = getattr(self.args, "eva_topk_ratio", 0.7)
        min_edge_len = 1e-4
        device = keyframe_meshes.device
        B, K, N, D = keyframe_meshes.shape
        keyframe_meshes_xyz = keyframe_meshes[..., :3]

        def _prep_edges(edges: torch.Tensor):
            if edges is None or edges.numel() == 0:
                return (
                    edges.new_zeros((0, 3)),
                    edges.new_zeros((0, 3)),
                    edges.new_zeros((0,)),
                )
            e = edges.reshape(-1, 2, 3)
            valid = torch.isfinite(e.view(e.size(0), -1)).all(dim=1)
            e = e[valid]
            if e.numel() == 0:
                return (
                    edges.new_zeros((0, 3)),
                    edges.new_zeros((0, 3)),
                    edges.new_zeros((0,)),
                )
            v = e[:, 1] - e[:, 0]
            l = torch.linalg.norm(v, dim=-1)
            keep = l > min_edge_len
            v, l = v[keep], l[keep]
            m = ((e[:, 1] + e[:, 0]) * 0.5)[keep]
            if v.numel() == 0:
                return (
                    edges.new_zeros((0, 3)),
                    edges.new_zeros((0, 3)),
                    edges.new_zeros((0,)),
                )
            return v, m, l

        def _structure_tensor(vecs: torch.Tensor, w: torch.Tensor | None):
            if vecs.numel() == 0:
                return torch.eye(3, device=device, dtype=torch.float32)
            vecs = vecs.to(torch.float32)
            if (w is None) or (w.numel() == 0):
                return vecs.transpose(0, 1) @ vecs
            W = w / (w.sum() + 1e-8)
            return vecs.transpose(0, 1) @ (vecs * W.unsqueeze(-1))

        def _estimate_rt_from_edges(v_ref, m_ref, w_ref, v_tgt, m_tgt, w_tgt):
            # Eigendecomposition is kept in full precision to avoid half-precision failures.
            with torch.cuda.amp.autocast(enabled=False):
                S_ref = _structure_tensor(v_ref, w_ref).to(torch.float32)
                S_tgt = _structure_tensor(v_tgt, w_tgt).to(torch.float32)

                eval_r, evec_r = torch.linalg.eigh(S_ref)
                eval_t, evec_t = torch.linalg.eigh(S_tgt)

                def _is_iso(evals, thr_ratio=1.2, thr_min=1e-5):
                    return (evals.min() > thr_min) and ((evals.max() / evals.min()) < thr_ratio)

                if _is_iso(eval_r) or _is_iso(eval_t):
                    R = torch.eye(3, device=S_ref.device, dtype=torch.float32)
                else:
                    Ur = evec_r[:, torch.argsort(eval_r, descending=True)]
                    Vt = evec_t[:, torch.argsort(eval_t, descending=True)]
                    R = Vt @ Ur.T
                    if torch.linalg.det(R) < 0:
                        Vt = Vt.clone()
                        Vt[:, -1] *= -1
                        R = Vt @ Ur.T

                if m_ref.numel() == 0 or m_tgt.numel() == 0:
                    t = torch.zeros(3, device=S_ref.device, dtype=torch.float32)
                else:
                    wr = (
                        (w_ref / (w_ref.sum() + 1e-8)).unsqueeze(1)
                        if (w_ref is not None and w_ref.numel() > 0)
                        else None
                    )
                    wt = (
                        (w_tgt / (w_tgt.sum() + 1e-8)).unsqueeze(1)
                        if (w_tgt is not None and w_tgt.numel() > 0)
                        else None
                    )
                    mu_ref = (wr * m_ref).sum(dim=0) if wr is not None else m_ref.mean(dim=0)
                    mu_tgt = (wt * m_tgt).sum(dim=0) if wt is not None else m_tgt.mean(dim=0)
                    t = mu_tgt - (R @ mu_ref)

            return R, t

        def _normalize(v):
            return v / (torch.linalg.norm(v, dim=-1, keepdim=True) + 1e-8)

        def _loss_triplet(ref_edges, tgt_edges):
            v_ref, m_ref, l_ref = _prep_edges(ref_edges)
            v_tgt, m_tgt, l_tgt = _prep_edges(tgt_edges)
            if v_ref.numel() == 0 or v_tgt.numel() == 0:
                zero = torch.tensor(0.0, device=device)
                return zero, zero, zero

            R, t = _estimate_rt_from_edges(v_ref, m_ref, l_ref.clone(), v_tgt, m_tgt, l_tgt.clone())

            v_ref_u_rot = (R @ _normalize(v_ref.to(torch.float32)).transpose(0, 1)).transpose(0, 1)
            v_tgt_u = _normalize(v_tgt.to(torch.float32))

            kr = max(1, int(v_ref_u_rot.size(0) * topk_ratio))
            kt = max(1, int(v_tgt_u.size(0) * topk_ratio))
            A = v_ref_u_rot[torch.topk(l_ref, k=min(kr, l_ref.size(0))).indices]
            B = v_tgt_u[torch.topk(l_tgt, k=min(kt, l_tgt.size(0))).indices]

            sim = (A @ B.transpose(0, 1)).clamp(-1.0, 1.0)
            L_dir = 0.5 * (
                1.0 - sim.max(dim=1).values.mean()
                + 1.0 - sim.max(dim=0).values.mean()
            )

            la, lb = torch.sort(l_ref)[0], torch.sort(l_tgt)[0]
            k = min(la.numel(), lb.numel())
            L_len = torch.mean((la[:k] - lb[:k]) ** 2) if k > 0 else torch.tensor(0.0, device=device)

            Pr, Pt = ref_edges.reshape(-1, 3), tgt_edges.reshape(-1, 3)
            Pr = Pr[torch.isfinite(Pr).all(dim=1)]
            Pt = Pt[torch.isfinite(Pt).all(dim=1)]
            if Pr.numel() == 0 or Pt.numel() == 0:
                L_ch = torch.tensor(0.0, device=device)
            else:
                Pr_aligned = (Pr @ R.transpose(0, 1)) + t
                # torch.cdist is O(MN), which is acceptable here because skeletons
                # contain far fewer points than dense point clouds.
                dist = torch.cdist(Pr_aligned.to(torch.float32), Pt.to(torch.float32), p=2)
                L_ch = 0.5 * (
                    (dist.min(dim=1).values ** 2).mean()
                    + (dist.min(dim=0).values ** 2).mean()
                )

            return L_dir, L_len, L_ch

        ref_tokens = anchor_labels.to(device).clone()
        for tok in (self.bos_token_id, self.eos_token_id, self.pad_token_id):
            ref_tokens[ref_tokens == tok] = self.pad_id
        ref_tokens = torch.where(ref_tokens != self.pad_id, ref_tokens - 3, ref_tokens)
        ref_edges = self.detokenize_joint_token(ref_tokens)

        L_dir_total = torch.tensor(0.0, device=device)
        L_len_total = torch.tensor(0.0, device=device)
        L_ch_total = torch.tensor(0.0, device=device)
        num_pairs = 0

        if K > 1:
            meshes_others_xyz = keyframe_meshes_xyz[:, 1:, :, :].reshape(-1, N, 3)

            if D < 6:
                zeros_for_normals = torch.zeros_like(meshes_others_xyz)
                meshes_others_for_gen = torch.cat([meshes_others_xyz, zeros_for_normals], dim=-1)
            else:
                meshes_others_for_gen = keyframe_meshes[:, 1:, :, :].reshape(-1, N, D)

            tgt_edges_all = self.generate(meshes_others_for_gen, deterministic=True)

            if tgt_edges_all is not None and tgt_edges_all.numel() > 0:
                tgt_edges_all = tgt_edges_all.view(B, K - 1, -1, 2, 3)
                for b in range(B):
                    for k in range(K - 1):
                        Ld, Ll, Lc = _loss_triplet(ref_edges[b], tgt_edges_all[b, k])
                        if torch.isfinite(Ld + Ll + Lc):
                            L_dir_total = L_dir_total + Ld
                            L_len_total = L_len_total + Ll
                            L_ch_total = L_ch_total + Lc
                            num_pairs += 1

        if num_pairs > 0:
            L_dir_avg = L_dir_total / num_pairs
            L_len_avg = L_len_total / num_pairs
            L_ch_avg = L_ch_total / num_pairs
        else:
            L_dir_avg = torch.tensor(0.0, device=device)
            L_len_avg = torch.tensor(0.0, device=device)
            L_ch_avg = torch.tensor(0.0, device=device)

        L_consistency = dir_w * L_dir_avg + len_w * L_len_avg + chamfer_w * L_ch_avg
        details = {
            "pairs": num_pairs,
            "L_dir": L_dir_avg.detach(),
            "L_len": L_len_avg.detach(),
            "L_ch": L_ch_avg.detach(),
        }
        return L_consistency, details

    def train_finetune_step(
        self,
        data_dict: dict,
        *,
        compute_selection_score: bool = False,
    ) -> dict:
        """
        Compute the weighted cross-pose token-distillation objective.

        The anchor frame uses standard teacher-forced supervision. Each
        non-anchor keyframe is additionally teacher-forced with the anchor
        token sequence. When ``compute_selection_score`` is enabled during
        validation, freely decoded geometry is added only to the detached
        checkpoint-selection score and never to the training gradient.
        """
        keyframe_meshes = data_dict["keyframe_meshes"]      # (B, K, N, 6)
        anchor_input_ids = data_dict["anchor_input_ids"]    # (B, L)
        anchor_labels = data_dict["anchor_labels"]          # (B, L)
        device = keyframe_meshes.device
        B, K = keyframe_meshes.size(0), keyframe_meshes.size(1)

        ignore_idx = -100
        # Keep per-token losses so that parent-index positions can be re-weighted explicitly.
        loss_fct = nn.CrossEntropyLoss(ignore_index=ignore_idx, reduction='none')

        # Special tokens are removed from the supervision target.
        loss_labels = anchor_labels.clone()
        loss_labels[loss_labels < 3] = ignore_idx

        # Attention mask covers both the conditional prefix and the token sequence.
        attn_tok = (anchor_input_ids != self.pad_token_id).long()
        attn_cond = torch.ones((B, self.cond_length), device=device, dtype=torch.long)
        attn_full = torch.cat([attn_cond, attn_tok], dim=1)

        # Part 1: anchor-frame teacher forcing.
        mesh_f0 = keyframe_meshes[:, 0]
        feat_f0 = self.point_encoder.encode_latents(mesh_f0)
        embeds_f0 = self.process_point_feature(point_feature=feat_f0)

        out0 = self.transformer(
            inputs_embeds=embeds_f0,
            input_ids=anchor_input_ids,
            attention_mask=attn_full,
            use_cache=False,
        )
        logits0 = out0.logits[:, self.cond_length - 1: -1, :]

        ce0 = loss_fct(logits0.reshape(-1, logits0.size(-1)), loss_labels.reshape(-1))
        valid_mask0 = (loss_labels != ignore_idx).reshape(-1)
        L_original = ce0[valid_mask0].mean() if valid_mask0.any() else ce0.sum() * 0.0

        # Part 1.1: increase the weight of parent-index tokens, following the
        # (x, y, z, parent) grouping used by the joint-token formulation.
        L_seq = anchor_labels.size(1)
        pos_idx = torch.arange(L_seq, device=device)
        parent_pos_mask_1xL = (pos_idx % 4 == 3).float().unsqueeze(0)
        parent_pos_mask = parent_pos_mask_1xL.expand(B, -1)

        parent_weight = getattr(self.args, "parent_ce_weight", 5.0)
        w_anchor = torch.ones_like(loss_labels, dtype=torch.float32)
        w_anchor = w_anchor + (parent_weight - 1.0) * parent_pos_mask
        w_anchor = w_anchor.reshape(-1)

        L_original_weighted = (
            (ce0 * w_anchor)[valid_mask0].sum() / (w_anchor[valid_mask0].sum() + 1e-9)
        ) if valid_mask0.any() else ce0.sum() * 0.0

        # Part 2: weighted token distillation on non-anchor support poses.
        lambda_cross = getattr(
            self.args,
            "lambda_cross",
            getattr(self.args, "lambda_token_consistency", 1.0),
        )
        max_frames = getattr(self.args, "max_frames_token_cons", 0)

        L_token_cons_list = []
        if K > 1:
            frames_idx = list(range(1, K))
            if (
                (max_frames is not None)
                and isinstance(max_frames, int)
                and (max_frames > 0)
                and (max_frames < len(frames_idx))
            ):
                # Uniform subsampling keeps the computation deterministic and reproducible.
                step = max(1, len(frames_idx) // max_frames)
                frames_idx = frames_idx[0:len(frames_idx):step][:max_frames]

            for k in frames_idx:
                mesh_fk = keyframe_meshes[:, k]
                feat_fk = self.point_encoder.encode_latents(mesh_fk)
                embeds_fk = self.process_point_feature(point_feature=feat_fk)

                outk = self.transformer(
                    inputs_embeds=embeds_fk,
                    input_ids=anchor_input_ids,
                    attention_mask=attn_full,
                    use_cache=False,
                )
                logitsk = outk.logits[:, self.cond_length - 1: -1, :]

                cek = loss_fct(logitsk.reshape(-1, logitsk.size(-1)), loss_labels.reshape(-1))
                valid_maskk = (loss_labels != ignore_idx).reshape(-1)

                w_k = torch.ones_like(loss_labels, dtype=torch.float32)
                w_k = w_k + (parent_weight - 1.0) * parent_pos_mask
                w_k = w_k.reshape(-1)

                L_tok_k = (
                    (cek * w_k)[valid_maskk].sum() / (w_k[valid_maskk].sum() + 1e-9)
                ) if valid_maskk.any() else cek.sum() * 0.0
                L_token_cons_list.append(L_tok_k)

        L_token_consistency = (
            torch.stack(L_token_cons_list).mean()
            if len(L_token_cons_list) > 0
            else logits0.sum() * 0.0
        )

        training_loss = L_original_weighted + lambda_cross * L_token_consistency

        # Hard decoding is validation-only. The resulting score is detached and
        # is used to rank fixed-budget epoch checkpoints.
        zero = training_loss.detach().new_zeros(())
        geometry_score = zero
        cons_details = {
            "pairs": 0,
            "L_dir": zero,
            "L_len": zero,
            "L_ch": zero,
        }
        if compute_selection_score:
            geometry_score, cons_details = self.decoded_geometry_score_against_anchor(
                keyframe_meshes=keyframe_meshes,
                anchor_labels=anchor_labels,
            )

        lambda_sel = getattr(
            self.args,
            "lambda_sel",
            getattr(self.args, "lambda_consistency", 0.5),
        )
        selection_score = training_loss.detach() + lambda_sel * geometry_score.detach()

        return_dict = {
            "loss": training_loss,
            "selection_score": selection_score,
            "loss_original": L_original.detach(),
            "loss_original_weighted": L_original_weighted.detach(),
            "geometry_score": geometry_score.detach(),
            "loss_consistency": (lambda_sel * geometry_score).detach(),
            "loss_token_consistency": (lambda_cross * L_token_consistency).detach(),
            "parent_ce_weight": torch.tensor(parent_weight, device=device),
            "lambda_sel": torch.tensor(lambda_sel, device=device),
            "lambda_cross": torch.tensor(lambda_cross, device=device),
        }
        return_dict.update(cons_details)
        return return_dict

    def forward(self, input_tensor, is_eval=False):
        if is_eval:
            return self.generate(input_tensor)
        return self.train_finetune_step(input_tensor)
