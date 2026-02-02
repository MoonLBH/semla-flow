import argparse
from dataclasses import dataclass
from pathlib import Path

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
from rdkit.Chem import QED

import semlaflow.scriptutil as util
import semlaflow.util.functional as smolF
import semlaflow.util.rdkit as smolRD
from semlaflow.data.datasets import GeometricDataset
from semlaflow.data.interpolate import GeometricInterpolant, GeometricNoiseSampler
from semlaflow.models.fm import Integrator, MolecularCFM
from semlaflow.models.semla import EquiInvDynamics, SemlaGenerator
from semlaflow.util.molrepr import GeometricMolBatch

DEFAULT_BATCH_SIZE = 64
DEFAULT_LR = 1e-4
DEFAULT_STEPS = 1000
DEFAULT_L2_WEIGHT = 0.1
DEFAULT_GRAD_CLIP = 1.0
DEFAULT_INTEGRATION_STEPS = 100
DEFAULT_ODE_SAMPLING_STRATEGY = "log"


@dataclass
class BetaBandit:
    sizes: list[int]
    alpha: torch.Tensor
    beta: torch.Tensor

    @classmethod
    def create(cls, sizes: list[int], alpha: float = 1.0, beta: float = 1.0) -> "BetaBandit":
        alpha_tensor = torch.full((len(sizes),), alpha, dtype=torch.float32)
        beta_tensor = torch.full((len(sizes),), beta, dtype=torch.float32)
        return cls(sizes=sizes, alpha=alpha_tensor, beta=beta_tensor)

    def sample(self) -> tuple[int, int]:
        samples = torch.distributions.Beta(self.alpha, self.beta).sample()
        index = torch.argmax(samples).item()
        return self.sizes[index], index

    def update(self, index: int, reward_mean: float) -> None:
        self.alpha[index] += reward_mean
        self.beta[index] += 1.0 - reward_mean


def load_model(args, vocab):
    checkpoint = torch.load(args.ckpt_path, map_location="cpu")
    hparams = checkpoint["hyper_parameters"]

    hparams["compile_model"] = False
    hparams["integration-steps"] = args.integration_steps
    hparams["sampling_strategy"] = args.ode_sampling_strategy

    n_bond_types = util.get_n_bond_types(hparams["integration-type-strategy"])

    if hparams.get("architecture") is None:
        hparams["architecture"] = "semla"

    if hparams["architecture"] == "semla":
        dynamics = EquiInvDynamics(
            hparams["d_model"],
            hparams["d_message"],
            hparams["n_coord_sets"],
            hparams["n_layers"],
            n_attn_heads=hparams["n_attn_heads"],
            d_message_hidden=hparams["d_message_hidden"],
            d_edge=hparams["d_edge"],
            self_cond=hparams["self_cond"],
            coord_norm=hparams["coord_norm"],
        )
        egnn_gen = SemlaGenerator(
            hparams["d_model"],
            dynamics,
            vocab.size,
            hparams["n_atom_feats"],
            d_edge=hparams["d_edge"],
            n_edge_types=n_bond_types,
            self_cond=hparams["self_cond"],
            size_emb=hparams["size_emb"],
            max_atoms=hparams["max_atoms"],
        )

    elif hparams["architecture"] == "eqgat":
        from semlaflow.models.eqgat import EqgatGenerator

        egnn_gen = EqgatGenerator(
            hparams["d_model"],
            hparams["n_layers"],
            hparams["n_equi_feats"],
            vocab.size,
            hparams["n_atom_feats"],
            hparams["d_edge"],
            hparams["n_edge_types"],
        )

    elif hparams["architecture"] == "egnn":
        from semlaflow.models.egnn import VanillaEgnnGenerator

        n_layers = args.n_layers if hparams.get("n_layers") is None else hparams["n_layers"]
        if n_layers is None:
            raise ValueError("No hparam for n_layers was saved, use script arg to provide n_layers")

        egnn_gen = VanillaEgnnGenerator(
            hparams["d_model"],
            n_layers,
            vocab.size,
            hparams["n_atom_feats"],
            d_edge=hparams["d_edge"],
            n_edge_types=n_bond_types,
        )

    else:
        raise ValueError("Unknown architecture hyperparameter.")

    type_mask_index = vocab.indices_from_tokens(["<MASK>"])[0] if hparams["integration-type-strategy"] == "mask" else None
    bond_mask_index = util.BOND_MASK_INDEX if hparams["integration-bond-strategy"] == "mask" else None

    integrator = Integrator(
        args.integration_steps,
        type_strategy=hparams["integration-type-strategy"],
        bond_strategy=hparams["integration-bond-strategy"],
        type_mask_index=type_mask_index,
        bond_mask_index=bond_mask_index,
        cat_noise_level=args.cat_sampling_noise_level,
    )

    model = MolecularCFM.load_from_checkpoint(
        args.ckpt_path,
        gen=egnn_gen,
        vocab=vocab,
        integrator=integrator,
        type_mask_index=type_mask_index,
        bond_mask_index=bond_mask_index,
        **hparams,
    )
    return model, hparams


def load_size_candidates(args) -> list[int]:
    dataset_path = Path(args.data_path) / f"{args.dataset_split}.smol"
    dataset = GeometricDataset.load(dataset_path)
    sizes = sorted(set(dataset.lengths))

    if args.min_atoms is not None:
        sizes = [size for size in sizes if size >= args.min_atoms]
    if args.max_atoms is not None:
        sizes = [size for size in sizes if size <= args.max_atoms]

    if not sizes:
        raise ValueError("No candidate molecule sizes remain after applying filters.")

    return sizes


def build_interpolant(hparams, vocab, n_bond_types):
    type_mask_index = vocab.indices_from_tokens(["<MASK>"])[0]
    bond_mask_index = util.BOND_MASK_INDEX

    type_interpolation = hparams.get("train-type-interpolation", "unmask")
    bond_interpolation = hparams.get("train-bond-interpolation", "unmask")
    type_noise = hparams.get("train-prior-type-noise", "uniform-sample")
    bond_noise = hparams.get("train-prior-bond-noise", "uniform-sample")
    coord_noise_std = hparams.get("train-coord-noise-std", 0.0)
    type_dist_temp = hparams.get("train-type-dist-temp", 1.0)
    equivariant_ot = hparams.get("train-equivariant-ot", False)
    batch_ot = hparams.get("train-batch-ot", False)
    time_alpha = hparams.get("train-time-alpha", 1.0)
    time_beta = hparams.get("train-time-beta", 1.0)
    fixed_time = hparams.get("train-fixed-interpolation-time")

    prior_sampler = GeometricNoiseSampler(
        vocab.size,
        n_bond_types,
        coord_noise="gaussian",
        type_noise=type_noise,
        bond_noise=bond_noise,
        scale_ot=hparams.get("train-prior-noise-scale-ot", False),
        zero_com=True,
        type_mask_index=type_mask_index,
        bond_mask_index=bond_mask_index,
    )

    interpolant = GeometricInterpolant(
        prior_sampler,
        coord_interpolation="linear",
        type_interpolation=type_interpolation,
        bond_interpolation=bond_interpolation,
        coord_noise_std=coord_noise_std,
        type_dist_temp=type_dist_temp,
        equivariant_ot=equivariant_ot,
        batch_ot=batch_ot,
        time_alpha=time_alpha,
        time_beta=time_beta,
        fixed_time=fixed_time,
    )
    return prior_sampler, interpolant


def interpolate_pairs(interpolant, from_mols, to_mols):
    if interpolant.batch_ot:
        from_mols = [mol.zero_com() for mol in from_mols]
        to_mols = [mol.zero_com() for mol in to_mols]
        from_mols = interpolant._ot_map(from_mols, to_mols)
    else:
        from_mols = [interpolant._match_mols(from_mol, to_mol) for from_mol, to_mol in zip(from_mols, to_mols)]

    if interpolant.fixed_time is not None:
        times = torch.tensor([interpolant.fixed_time] * len(to_mols))
    else:
        times = interpolant.time_dist.sample((len(to_mols),))

    interp_mols = [
        interpolant._interpolate_mol(from_mol, to_mol, t)
        for from_mol, to_mol, t in zip(from_mols, to_mols, times.tolist())
    ]
    return from_mols, to_mols, interp_mols, times


def batch_to_dict(smol_batch: GeometricMolBatch) -> dict[str, torch.Tensor]:
    coords = smol_batch.coords.float()
    atomics = smol_batch.atomics.float()
    bonds = smol_batch.adjacency.float()
    charges = smol_batch.charges.long()
    mask = smol_batch.mask.long()

    if charges is not None:
        n_charges = len(smolRD.CHARGE_IDX_MAP.keys())
        charges = smolF.one_hot_encode_tensor(charges, n_charges)

    return {"coords": coords, "atomics": atomics, "bonds": bonds, "charges": charges, "mask": mask}


def predict_with_self_condition(model, interpolated, times):
    cond_batch = None
    if model.self_condition:
        cond_batch = {
            "coords": torch.zeros_like(interpolated["coords"]),
            "atomics": torch.zeros_like(interpolated["atomics"]),
            "bonds": torch.zeros_like(interpolated["bonds"]),
        }

        if torch.rand(1).item() > 0.5:
            with torch.no_grad():
                cond_coords, cond_types, cond_bonds, _ = model(
                    interpolated,
                    times,
                    training=True,
                    cond_batch=cond_batch,
                )
                cond_batch = {
                    "coords": cond_coords,
                    "atomics": F.softmax(cond_types, dim=-1),
                    "bonds": F.softmax(cond_bonds, dim=-1),
                }

    coords, types, bonds, charges = model(interpolated, times, training=True, cond_batch=cond_batch)
    return {"coords": coords, "atomics": types, "bonds": bonds, "charges": charges}


def compute_qed(model, generated):
    rdkit_mols = model.builder.mols_from_tensors(
        generated["coords"],
        generated["atomics"],
        generated["mask"],
        bond_dists=generated["bonds"],
        charge_dists=generated["charges"],
        sanitise=True,
    )
    scores = [QED.qed(mol) if mol is not None else 0.0 for mol in rdkit_mols]
    return torch.tensor(scores, device=generated["coords"].device, dtype=torch.float32)


def compute_l2_regularization(model, prior_state):
    total = 0.0
    total_params = 0
    for name, param in model.gen.named_parameters():
        total += (param - prior_state[name]).pow(2).sum()
        total_params += param.numel()
    return total / max(total_params, 1)


def save_checkpoint(save_path: Path, model: MolecularCFM) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "hyper_parameters": dict(model.hparams)}, save_path)


def main(args):
    torch.set_float32_matmul_precision("high")
    L.seed_everything(12345)
    util.disable_lib_stdout()
    util.configure_fs()

    vocab = util.build_vocab()
    model, hparams = load_model(args, vocab)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(device)
    model.train()

    prior_state = {name: param.detach().clone() for name, param in model.gen.named_parameters()}

    sizes = load_size_candidates(args)
    bandit = BetaBandit.create(sizes, alpha=args.bandit_alpha, beta=args.bandit_beta)

    n_bond_types = util.get_n_bond_types(hparams["integration-type-strategy"])
    prior_sampler, interpolant = build_interpolant(hparams, vocab, n_bond_types)

    optimizer = torch.optim.Adam(model.gen.parameters(), lr=args.lr, amsgrad=True, foreach=True)

    for step in range(1, args.steps + 1):
        mol_size, size_index = bandit.sample()
        prior_batch = prior_sampler.sample_batch([mol_size] * args.batch_size).to_device(device)
        prior_dict = batch_to_dict(prior_batch)
        prior_dict = {key: value.to(device) for key, value in prior_dict.items()}

        generated = model._generate(prior_dict, args.integration_steps, args.ode_sampling_strategy)
        qed_rewards = compute_qed(model, generated)
        baseline = qed_rewards.mean()
        centered_rewards = qed_rewards - baseline

        bandit.update(size_index, qed_rewards.mean().item())

        generated_for_loss = {**generated}
        generated_for_loss["coords"] = generated_for_loss["coords"] / model.coord_scale
        generated_smol = model.builder.smol_from_tensors(
            generated_for_loss["coords"],
            generated_for_loss["atomics"],
            generated_for_loss["mask"],
            bond_dists=generated_for_loss["bonds"],
            charge_dists=generated_for_loss["charges"],
        )

        from_mols = prior_batch.to_list()
        _, to_mols, interp_mols, times = interpolate_pairs(interpolant, from_mols, generated_smol)
        data_batch = GeometricMolBatch.from_list(to_mols)
        interp_batch = GeometricMolBatch.from_list(interp_mols)

        data_dict = {key: value.to(device) for key, value in batch_to_dict(data_batch).items()}
        interp_dict = {key: value.to(device) for key, value in batch_to_dict(interp_batch).items()}
        times = times.to(device=device, dtype=torch.float32)

        predicted = predict_with_self_condition(model, interp_dict, times)
        base_loss, components = model.loss_per_sample(data_dict, interp_dict, predicted)
        rl_loss = (centered_rewards * base_loss).mean()
        l2_reg = compute_l2_regularization(model, prior_state)
        loss = rl_loss + (args.l2_weight * l2_reg)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.gen.parameters(), args.grad_clip)
        optimizer.step()

        if step % args.log_every == 0 or step == 1:
            mean_qed = qed_rewards.mean().item()
            print(
                "step",
                step,
                "size",
                mol_size,
                "qed",
                f"{mean_qed:.4f}",
                "loss",
                f"{loss.item():.4f}",
                "rl",
                f"{rl_loss.item():.4f}",
                "l2",
                f"{l2_reg.item():.6f}",
            )

            for name, values in components.items():
                print(f"  {name}: {values.mean().item():.4f}")

        if args.save_every and step % args.save_every == 0:
            save_checkpoint(Path(args.save_dir) / f"{args.save_prefix}-step{step}.ckpt", model)

    save_checkpoint(Path(args.save_dir) / f"{args.save_prefix}-final.ckpt", model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--device", type=str, default=None)

    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--grad_clip", type=float, default=DEFAULT_GRAD_CLIP)
    parser.add_argument("--l2_weight", type=float, default=DEFAULT_L2_WEIGHT)

    parser.add_argument("--integration_steps", type=int, default=DEFAULT_INTEGRATION_STEPS)
    parser.add_argument("--ode_sampling_strategy", type=str, default=DEFAULT_ODE_SAMPLING_STRATEGY)
    parser.add_argument("--cat_sampling_noise_level", type=float, default=1.0)

    parser.add_argument("--bandit_alpha", type=float, default=1.0)
    parser.add_argument("--bandit_beta", type=float, default=1.0)
    parser.add_argument("--min_atoms", type=int, default=None)
    parser.add_argument("--max_atoms", type=int, default=None)

    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=0)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--save_prefix", type=str, default="semlaflow-rl-qed")

    parser.add_argument("--n_layers", type=int, default=None)

    args = parser.parse_args()
    main(args)
