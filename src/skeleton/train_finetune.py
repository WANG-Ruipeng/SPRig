"""
Skeleton post-training script for SPRig.

The optimization objective is weighted cross-pose token distillation from
one anchor target. A hard-decoded geometry score is computed without
gradients on validation data and is used only for checkpoint selection.
"""

import argparse
import gc
import os
import shutil

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from animation_dataset import AnimationSequenceDataset
from skeleton_gpt_finetune import SkeletonGPT


def get_args():
    parser = argparse.ArgumentParser("SkeletonGPT fine-tuning script")
    parser.add_argument('--train_npz_dir', required=True, type=str)
    parser.add_argument('--val_npz_dir', required=True, type=str)
    parser.add_argument('--pretrained_weights', required=True, type=str)
    parser.add_argument('--output_dir', required=True, type=str)
    parser.add_argument('--epochs', default=12, type=int)
    parser.add_argument('--learning_rate', default=3e-6, type=float)
    parser.add_argument(
        '--batch_size',
        default=20,
        type=int,
        help="Physical batch size before gradient accumulation.",
    )
    parser.add_argument('--log_every', default=100, type=int)
    parser.add_argument('--llm', default="facebook/opt-350m", type=str)
    parser.add_argument("--n_discrete_size", default=128, type=int)
    parser.add_argument("--n_max_bones", default=100, type=int)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument(
        '--accumulation_steps',
        default=1,
        type=int,
        help="Number of gradient accumulation steps.",
    )
    parser.add_argument(
        '--weight_decay',
        default=0.01,
        type=float,
        help="Weight decay used by AdamW.",
    )

    parser.add_argument(
        '--lambda_sel',
        '--lambda_consistency',
        dest='lambda_sel',
        default=0.5,
        type=float,
        help="Weight for the validation-only decoded-geometry selection score.",
    )
    parser.add_argument(
        '--lambda_dir',
        default=1.0,
        type=float,
        help="Weight for directional consistency in EVA.",
    )
    parser.add_argument(
        '--lambda_len',
        default=0.2,
        type=float,
        help="Weight for edge-length distribution consistency in EVA.",
    )
    parser.add_argument(
        '--lambda_chamfer',
        default=0.5,
        type=float,
        help="Weight for endpoint-set Chamfer consistency in EVA.",
    )
    parser.add_argument(
        '--eva_topk_ratio',
        default=0.7,
        type=float,
        help="Ratio of long edges used when estimating dominant directions in EVA.",
    )
    parser.add_argument(
        '--unfreeze_encoder',
        action='store_true',
        help="If set, fine-tune the point encoder together with the decoder.",
    )
    parser.add_argument(
        '--parent_ce_weight',
        type=float,
        default=5.0,
        help="Relative weight assigned to parent-index tokens in cross-entropy.",
    )
    parser.add_argument(
        '--lambda_cross',
        '--lambda_token_consistency',
        dest='lambda_cross',
        type=float,
        default=1.0,
        help="Weight for cross-pose token distillation.",
    )
    parser.add_argument(
        '--max_frames_token_cons',
        type=int,
        default=0,
        help="Maximum number of non-anchor keyframes used for token consistency; 0 uses all.",
    )
    return parser.parse_args()


def custom_collate_fn(batch_list):
    """
    Collate a mini-batch of animation sequences.

    The fine-tuning objective uses only the anchor tokens and the stacked
    keyframe point clouds.
    """
    input_ids = [item['anchor_input_ids'] for item in batch_list]
    padded_input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids,
        batch_first=True,
        padding_value=2,
    )

    labels = [item['anchor_labels'] for item in batch_list]
    padded_labels = torch.nn.utils.rnn.pad_sequence(
        labels,
        batch_first=True,
        padding_value=2,
    )

    meshes = torch.stack([item['keyframe_meshes'] for item in batch_list])

    return {
        'keyframe_meshes': meshes,
        'anchor_input_ids': padded_input_ids,
        'anchor_labels': padded_labels,
    }


def main():
    args = get_args()

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda"

    os.makedirs(args.output_dir, exist_ok=True)

    checkpoint_dir = os.path.join(args.output_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    save_path = os.path.join(checkpoint_dir, "best_model.pth")

    log_dir = os.path.join(args.output_dir, 'logs')
    writer = SummaryWriter(log_dir=log_dir)

    # Release preparation (2026-11-10): use the explicit input paths.
    # The original script copied into and deleted one global Colab cache.
    # Keep preprocessing and losses unchanged; users may cache inputs themselves.
    for data_dir in (args.train_npz_dir, args.val_npz_dir):
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(data_dir)
    if not os.path.isfile(args.pretrained_weights):
        raise FileNotFoundError(args.pretrained_weights)

    train_dataset = AnimationSequenceDataset(dataset_dir=args.train_npz_dir)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        collate_fn=custom_collate_fn,
        shuffle=True,
        num_workers=4,
    )

    val_dataset = AnimationSequenceDataset(dataset_dir=args.val_npz_dir)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        collate_fn=custom_collate_fn,
        shuffle=False,
        num_workers=4,
    )

    model_args = argparse.Namespace(**vars(args))
    model_args.joint_token = True
    model_args.seq_shuffle = False
    model_args.batchsize_per_gpu = args.batch_size
    model_args.num_beams = 1
    model_args.do_sample = True
    model_args.top_k = 50
    model_args.top_p = 0.95

    model = SkeletonGPT(model_args).to(device)

    pkg = torch.load(args.pretrained_weights, map_location="cpu")
    pretrained_dict = pkg.get("model", pkg)
    model_dict = model.state_dict()
    mismatched_key = 'transformer.model.decoder.embed_positions.weight'
    if (
        mismatched_key in pretrained_dict
        and mismatched_key in model_dict
        and pretrained_dict[mismatched_key].shape != model_dict[mismatched_key].shape
    ):
        del pretrained_dict[mismatched_key]
    model.load_state_dict(pretrained_dict, strict=False)

    if args.unfreeze_encoder:
        params_to_update = model.parameters()
    else:
        params_to_update = []
        for name, param in model.named_parameters():
            if "point_encoder" in name:
                param.requires_grad = False
            else:
                params_to_update.append(param)

    optimizer = torch.optim.AdamW(
        params_to_update,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    if args.debug:
        train_dataloader = [next(iter(train_dataloader))]
        val_dataloader = [next(iter(val_dataloader))]
        args.epochs = 1

    best_selection_score = float('inf')
    global_step = 0

    for epoch in range(args.epochs):
        model.train()
        train_loss_total = 0.0
        optimizer.zero_grad()

        for i, batch in enumerate(train_dataloader):
            batch_on_device = {k: v.to(device) for k, v in batch.items()}

            with torch.cuda.amp.autocast(enabled=amp_enabled):
                loss_dict = model(batch_on_device, is_eval=False)
                loss = loss_dict['loss'] / args.accumulation_steps

            scaler.scale(loss).backward()

            if (i + 1) % args.accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_loss_total += loss.item() * args.accumulation_steps

            if global_step % args.log_every == 0 and (i + 1) % args.accumulation_steps == 0:
                writer.add_scalar('Loss/train_step', loss.item() * args.accumulation_steps, global_step)
                writer.add_scalar('Loss_components/original', loss_dict['loss_original'].item(), global_step)
                writer.add_scalar(
                    'Loss_components/cross_pose_token',
                    loss_dict['loss_token_consistency'].item(),
                    global_step,
                )

            global_step += 1

        model.eval()

        val_loss_total = 0.0
        val_ori_total = 0.0
        val_ori_w_total = 0.0
        val_cons_total = 0.0
        val_tok_cons_total = 0.0
        val_selection_total = 0.0

        with torch.no_grad():
            for batch in val_dataloader:
                batch_on_device = {k: v.to(device) for k, v in batch.items()}

                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    loss_dict = model.train_finetune_step(
                        batch_on_device,
                        compute_selection_score=True,
                    )
                    val_loss_total += loss_dict['loss'].item()
                    val_ori_total += loss_dict['loss_original'].item()
                    val_ori_w_total += loss_dict['loss_original_weighted'].item()
                    val_cons_total += loss_dict['loss_consistency'].item()
                    val_tok_cons_total += loss_dict['loss_token_consistency'].item()
                    val_selection_total += loss_dict['selection_score'].item()

        num_train_batches = len(train_dataloader)
        num_val_batches = len(val_dataloader)

        avg_train_loss = train_loss_total / num_train_batches if num_train_batches > 0 else 0.0
        avg_val_loss = val_loss_total / num_val_batches if num_val_batches > 0 else 0.0
        avg_val_ori = val_ori_total / num_val_batches if num_val_batches > 0 else 0.0
        avg_val_ori_w = val_ori_w_total / num_val_batches if num_val_batches > 0 else 0.0
        avg_val_cons = val_cons_total / num_val_batches if num_val_batches > 0 else 0.0
        avg_val_tok_cons = val_tok_cons_total / num_val_batches if num_val_batches > 0 else 0.0
        avg_selection_score = (
            val_selection_total / num_val_batches
            if num_val_batches > 0
            else float('inf')
        )

        writer.add_scalar('Loss/train_epoch_avg', avg_train_loss, epoch)
        writer.add_scalar('Val_Loss/total', avg_val_loss, epoch)
        writer.add_scalar('Val_Loss_Components/original', avg_val_ori, epoch)
        writer.add_scalar('Val_Loss_Components/original_weighted', avg_val_ori_w, epoch)
        writer.add_scalar('Val_Score_Components/decoded_geometry_weighted', avg_val_cons, epoch)
        writer.add_scalar('Val_Loss_Components/cross_pose_token', avg_val_tok_cons, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)

        writer.add_scalar('Val_Loss/checkpoint_selection_score', avg_selection_score, epoch)

        if avg_selection_score < best_selection_score:
            best_selection_score = avg_selection_score
            torch.save(model.state_dict(), save_path)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    writer.close()


if __name__ == '__main__':
    main()
