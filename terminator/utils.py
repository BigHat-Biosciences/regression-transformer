import logging
import os
import sys

import numpy as np
import rdkit.rdBase as rkrb
import rdkit.RDLogger as rkl
import torch

from typing import List, Dict, Tuple, cast
from rdkit import Chem

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def get_device():
    return torch.device("cuda" if cuda() else "cpu")


def cuda():
    return torch.cuda.is_available()


def map_tensor_dict(
    tensor_dict: Dict[str, torch.Tensor], device: torch.device
) -> Dict[str, torch.Tensor]:
    """
    Maps a dictionary of tensors to a specific device.

    Args:
        tensor_dict: A dictionary of tensors.
        device: The device to map the tensors to.

    Returns:
        A dictionary of tensors mapped to the device.
    """
    return {key: tensor.to(device) for key, tensor in tensor_dict.items()}


def device_claim(device = None) -> torch.device:
    """
    Satidfy a device claim.

    Args:
        device: device where the inference is running either as a dedicated class or
            a string. If not provided is inferred.

    Returns:
        torch.device: the claimed device or a default one.
    """
    if isinstance(device, str):
        device = torch.device(device)
    device = (
        get_device()
        if (device is None or not isinstance(device, torch.device))
        else device
    )
    return device


def get_substructure_indices(
    full_sequence: List[str], substructure: List[str]
) -> List[int]:
    """
    Args:
        full_sequence: A list of strings, each representing a token from the full sequence
        substructure: A list of strings, each representing a token from the substructure that
            is contained in the full sequence.

    Returns:
        A list of integers, corresponding to all the indices of the tokens in the full sequence
        that match the substructure.

    """
    substructure_indices: List = []
    for i in range(len(full_sequence)):
        if full_sequence[i] == substructure[0]:
            if full_sequence[i : i + len(substructure)] == substructure:
                substructure_indices.extend(range(i, i + len(substructure)))
    return substructure_indices


def filter_stubbed(
    property_sequences: Tuple[Tuple[str, str]], target: str, threshold: float = 0.5
) -> Tuple[Tuple[str, str]]:
    """
    Remove stub-like molecules that are substantially smaller than the target.

    Args:
        sequences: List of generated molecules.
        properties: Properties of the molecules. Only used to be returned after filtering.
        target: Seed molecule.
        threshold: Fraction of size of generated molecule compared to seed determining the
            threshold under which molecules are discarded. Defaults to 0.5.

    Returns:
        Tuple of tuples of length 2 with filtered, generated molecule and its properties.
    """

    seed = Chem.MolFromSmiles(target)  # type: ignore

    seed_atoms = len(list(seed.GetAtoms()))  # type: ignore
    seed_bonds = seed.GetNumBonds()  # type: ignore

    smis: List[str] = []
    props: List[str] = []
    for smi, prop in property_sequences:
        if smi == "":
            continue
        try:
            mol = Chem.MolFromSmiles(smi)  # type: ignore
            num_atoms = len(list(mol.GetAtoms()))  # type: ignore
            num_bonds = mol.GetNumBonds()  # type: ignore

            if num_atoms > (threshold * seed_atoms) and num_bonds > (
                threshold * seed_bonds
            ):
                smis.append(smi)
                props.append(prop)
        except Exception:
            continue

    successes = cast(Tuple[Tuple[str, str]], tuple(zip(smis, props)))
    return successes


def get_latest_checkpoint(model_path: str, must_contain: str = "best") -> str:
    """
    Given a path to the model folder it searches the latest saved checkpoint
    and returns the path to it.

    Args:
        model_path (str): Path to model folder. Has to contain folders called
            'checkpoint-best-STEP' and 'checkpoint-latest-STEP' where STEP is
            a positive integer.
        must_contain (str, optional): Subselect checkpoints that contain a
            certain query. Defaults to 'best'.

    Returns:
        str: Path to latest checkpoint
    """

    # Finding checkpoints
    checkpoints = [f for f in os.listdir(model_path) if f.startswith("checkpoint")]
    if must_contain is not None:
        checkpoints = list(filter(lambda x: must_contain in x, checkpoints))

    if len(checkpoints) == 0:
        logger.warning(
            f"No checkpoints found that contain {must_contain} in {model_path}."
        )
        # Relax criteria and retry
        next_try = "checkpoint" if must_contain != "checkpoint" else ""
        return get_latest_checkpoint(model_path, must_contain=next_try)

    # Sorting
    try:
        idx = np.argsort([int(c.split("-")[-1]) for c in checkpoints])[-1]
    except ValueError:
        raise ValueError(f"Checkpoints dont seem to follow format: {checkpoints}.")

    return os.path.join(model_path, checkpoints[idx])


def disable_rdkit_logging():
    """
    Disables RDKit whiny logging.
    """
    logger = rkl.logger()
    logger.setLevel(rkl.ERROR)
    rkrb.DisableLog("rdApp.error")


def find_safe_path(path: str) -> str:
    """Method to find a safe path that does not exist yet.

    Args:
        path (str): Desired path.

    Returns:
        str: Non existing path.
    """
    safe_path = path
    c = 0
    while os.path.exists(safe_path):
        c += 1
        safe_path = ".".join(
            [
                s if i != path.count(".") - 1 else f"{s}_v{c}"
                for i, s in enumerate(path.split("."))
            ]
        )
    return safe_path
