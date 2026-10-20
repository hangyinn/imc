import sys
import types

import numpy as np
import pandas as pd
import pytest

from imc.segmentation import (
    align_cell_and_nuclear_masks,
    deepcell_postprocess_both_compartments,
    deepcell_segment,
    deepcell_segment_probabilities,
    percentile_normalize,
    prepare_stack,
    resize_labels,
    segment_roi,
    to_deepcell_input,
)


def disk(shape, center, radius):
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    return (yy - center[0]) ** 2 + (xx - center[1]) ** 2 <= radius**2


def nuclei_per_cell(mask):
    cell, nuc = mask[..., 0], mask[..., 1]
    return {
        c: set(np.unique(nuc[cell == c])) - {0} for c in np.unique(cell) if c != 0
    }


def check_pairing(mask):
    """Nuclei lie in their cell, carry its label, and labels are consecutive."""
    cell, nuc = mask[..., 0], mask[..., 1]
    assert (cell[nuc > 0] == nuc[nuc > 0]).all()
    labels = np.unique(cell)
    labels = labels[labels > 0]
    assert (labels == np.arange(1, len(labels) + 1)).all()
    assert all(len(v) <= 1 for v in nuclei_per_cell(mask).values())


# Input images
LABELS = pd.Series(
    [
        "80ArAr(ArAr80)",
        "DNA1(Ir191)",
        "DNA2(Ir193)",
        "CD45(Sm152)",
        "CD3(Er170)",
        "Ki67(Er168)",
    ]
)


def make_stack():
    rng = np.random.default_rng(0)
    shape = (64, 64)
    nucleus = disk(shape, (32, 32), 6)
    membrane = disk(shape, (32, 32), 10) & ~disk(shape, (32, 32), 8)
    stack = rng.poisson(1, (len(LABELS),) + shape).astype(float)
    stack[1][nucleus] += 50
    stack[2][nucleus] += 50
    stack[3][membrane] += 50
    stack[4][membrane] += 30
    stack[5][nucleus] += 50
    stack[0] += 20
    return stack, nucleus, membrane


def test_membrane_image_uses_membrane_channels():
    stack, nucleus, membrane = make_stack()
    cyto = prepare_stack(stack, LABELS, "cytoplasm")
    assert cyto[membrane].mean() > 2 * cyto[nucleus].mean()


def test_membrane_channels_can_be_chosen():
    stack, _, _ = make_stack()
    cyto = prepare_stack(
        stack, LABELS, "cytoplasm", membrane_channels=["CD3"], normalization="percentile"
    )
    np.testing.assert_allclose(cyto, percentile_normalize(stack[4]))
    with pytest.raises(ValueError):
        prepare_stack(stack, LABELS, "cytoplasm", membrane_channels=["CD20"])


def test_both_compartments_nuclear_first():
    stack, nucleus, membrane = make_stack()
    both = prepare_stack(stack, LABELS, "both")
    assert both.shape == (2, 64, 64)
    assert both[0][nucleus].mean() > both[0][membrane].mean()
    assert both[1][membrane].mean() > both[1][nucleus].mean()


def test_nuclear_image_unchanged_with_exclusion():
    stack, _, _ = make_stack()
    exclude = pd.Series([True, False, False, False, False, False], index=LABELS.values)
    a = prepare_stack(stack, LABELS, "nuclear")
    b = prepare_stack(stack, LABELS, "nuclear", channel_exclude=exclude)
    np.testing.assert_allclose(a, b)


def test_deepcell_input_keeps_resolution():
    image = np.random.rand(2, 300, 500)
    out = to_deepcell_input(image)
    assert out.shape == (1, 300, 500, 2)
    np.testing.assert_array_equal(out[0, ..., 1], image[1])


def test_label_resize_keeps_label_values():
    mask = np.zeros((50, 50), dtype=np.uint32)
    mask[5:20, 5:20] = 3
    mask[20:40, 10:45] = 7
    assert set(np.unique(resize_labels(mask, (100, 100)))) == {0, 3, 7}
    assert set(np.unique(resize_labels(mask, (23, 31)))) <= {0, 3, 7}


# DeepCell calls
@pytest.fixture
def deepcell_calls(monkeypatch):
    calls = list()

    class FakeApplication:
        def predict(self, image, image_mpp=None, compartment="whole-cell", **kwargs):
            calls.append(dict(shape=image.shape, image_mpp=image_mpp))
            n = 2 if compartment == "both" else 1
            return np.ones(image.shape[:3] + (n,), dtype=np.int32)

    applications = types.ModuleType("deepcell.applications")
    applications.MultiplexSegmentation = FakeApplication
    applications.NuclearSegmentation = FakeApplication
    applications.CytoplasmSegmentation = FakeApplication
    deepcell = types.ModuleType("deepcell")
    deepcell.applications = applications
    monkeypatch.setitem(sys.modules, "deepcell", deepcell)
    monkeypatch.setitem(sys.modules, "deepcell.applications", applications)
    return calls


def test_deepcell_gets_full_resolution_and_pixel_size(deepcell_calls):
    image = np.random.rand(2, 300, 500)
    pred = deepcell_segment(to_deepcell_input(image), compartment="both", image_mpp=1.0)
    assert deepcell_calls[0] == dict(shape=(1, 300, 500, 2), image_mpp=1.0)
    assert pred.shape == (300, 500, 2)


def test_probabilities_get_pixel_size(deepcell_calls):
    deepcell_segment_probabilities(np.random.rand(120, 80, 3), "both", image_mpp=1.0)
    assert deepcell_calls[0] == dict(shape=(1, 120, 80, 2), image_mpp=1.0)


class FakeROI:
    def __init__(self, stack, labels):
        self.stack = stack
        self.channel_labels = labels
        self.channel_exclude = pd.Series(False, index=labels.values)


def test_segment_roi_from_stack(deepcell_calls):
    stack, _, _ = make_stack()
    masks = segment_roi(
        FakeROI(stack, LABELS),
        model="deepcell",
        compartment="both",
        membrane_channels=["CD45", "CD3"],
        save=False,
        plot_segmentation=False,
    )
    assert deepcell_calls[0] == dict(shape=(1, 64, 64, 2), image_mpp=1.0)
    assert masks["cell"].shape == masks["nuclei"].shape == (64, 64)


# Pairing of cell and nuclear masks
SHAPE = (40, 60)


def two_nuclei_in_one_cell():
    cell = np.zeros(SHAPE, int)
    nuc = np.zeros(SHAPE, int)
    cell[5:35, 5:55] = 1
    nuc[disk(SHAPE, (20, 18), 5)] = 1
    nuc[disk(SHAPE, (20, 42), 5)] = 2
    return np.stack([cell, nuc], -1)


def nucleus_shared_by_two_cells(center_x=30):
    cell = np.zeros(SHAPE, int)
    nuc = np.zeros(SHAPE, int)
    cell[5:35, 5:30] = 1
    cell[5:35, 30:55] = 2
    nuc[disk(SHAPE, (20, center_x), 6)] = 1
    return np.stack([cell, nuc], -1)


def test_cell_with_two_nuclei_is_divided():
    mask = two_nuclei_in_one_cell()
    out = align_cell_and_nuclear_masks(mask)
    check_pairing(out)
    assert len(nuclei_per_cell(out)) == 2
    assert all(len(v) == 1 for v in nuclei_per_cell(out).values())
    assert ((out[..., 0] > 0) == (mask[..., 0] > 0)).all()
    # the border between the two cells lies between the nuclei
    assert out[20, 18, 0] != out[20, 42, 0]


def test_legacy_removes_cell_with_two_nuclei():
    out = deepcell_postprocess_both_compartments(two_nuclei_in_one_cell())
    assert out[..., 0].max() == 0


def test_shared_nucleus_is_divided():
    out = align_cell_and_nuclear_masks(nucleus_shared_by_two_cells())
    check_pairing(out)
    assert len(nuclei_per_cell(out)) == 2
    assert all(len(v) == 1 for v in nuclei_per_cell(out).values())


def test_legacy_removes_cells_sharing_a_nucleus():
    out = deepcell_postprocess_both_compartments(nucleus_shared_by_two_cells())
    assert out[..., 0].max() == 0


def test_dominant_nucleus_is_given_to_one_cell():
    mask = nucleus_shared_by_two_cells(center_x=26)
    nuc = mask[..., 1] == 1
    assert (mask[..., 0][nuc] == 1).mean() > 0.8
    out = align_cell_and_nuclear_masks(mask)
    check_pairing(out)
    assert len(np.unique(out[..., 0][nuc])) == 1
    assert (out[..., 1][nuc] > 0).all()


def test_small_nuclear_fragment_is_merged():
    cell = np.zeros(SHAPE, int)
    nuc = np.zeros(SHAPE, int)
    cell[5:35, 5:35] = 1
    nuc[disk(SHAPE, (20, 18), 7)] = 1
    nuc[20:22, 30:32] = 2
    out = align_cell_and_nuclear_masks(np.stack([cell, nuc], -1))
    check_pairing(out)
    assert out[..., 0].max() == 1


def test_cell_without_nucleus_is_kept_empty():
    cell = np.zeros(SHAPE, int)
    nuc = np.zeros(SHAPE, int)
    cell[5:20, 5:20] = 1
    cell[22:38, 30:50] = 2
    nuc[disk(SHAPE, (30, 40), 4)] = 1
    mask = np.stack([cell, nuc], -1)
    out = align_cell_and_nuclear_masks(mask)
    check_pairing(out)
    assert out[..., 0].max() == 2
    assert out[10, 10, 1] == 0
    legacy = deepcell_postprocess_both_compartments(mask)
    assert legacy[10, 10, 1] > 0


def test_nucleus_outside_cells_becomes_cell():
    cell = np.zeros(SHAPE, int)
    nuc = np.zeros(SHAPE, int)
    cell[5:20, 5:20] = 1
    nuc[disk(SHAPE, (12, 12), 4)] = 1
    nuc[disk(SHAPE, (30, 45), 4)] = 2
    out = align_cell_and_nuclear_masks(np.stack([cell, nuc], -1))
    check_pairing(out)
    assert out[..., 0].max() == 2
    new = out[30, 45, 0]
    assert new > 0 and out[30, 45, 1] == new
    assert (out[..., 0] == new).sum() > (nuc == 2).sum()


def test_gaps_filled_from_dna():
    cell = np.zeros(SHAPE, int)
    nuc = np.zeros(SHAPE, int)
    cell[5:20, 5:20] = 1
    nuc[disk(SHAPE, (12, 12), 4)] = 1
    dna = np.zeros(SHAPE)
    dna[disk(SHAPE, (12, 12), 4)] = 1
    dna[disk(SHAPE, (30, 45), 4)] = 1
    mask = np.stack([cell, nuc], -1)
    assert align_cell_and_nuclear_masks(mask)[..., 0].max() == 1
    out = align_cell_and_nuclear_masks(mask, nuclear_image=dna, fill_gaps=True)
    check_pairing(out)
    assert out[..., 0].max() == 2
    assert out[30, 45, 0] > 0 and out[30, 45, 1] == out[30, 45, 0]


def test_pairing_of_random_scene():
    from skimage.segmentation import watershed

    rng = np.random.default_rng(1)
    shape = (200, 200)
    points = rng.integers(5, 195, (120, 2))
    markers = np.zeros(shape, int)
    markers[tuple(points.T)] = np.arange(1, len(points) + 1)
    cell = watershed(np.zeros(shape), markers)
    cell[rng.random(shape) < 0.002] = 0
    nuc = np.zeros(shape, int)
    for i, (y, x) in enumerate(points + rng.integers(-3, 4, points.shape), 1):
        nuc[disk(shape, (y, x), rng.integers(2, 6))] = i
    cell[nuc == 7] = 0
    mask = np.stack([cell, nuc], -1)
    out = align_cell_and_nuclear_masks(mask)
    check_pairing(out)
    assert out[..., 0].max() >= len(np.unique(cell)) - 1 - 5


def test_segment_cli_options(monkeypatch):
    # the command line refuses to load on Windows; parsing does not depend on it
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delitem(sys.modules, "imc.scripts", raising=False)
    from imc.scripts import build_cli

    args = build_cli("segment").parse_args(
        [
            "--model",
            "deepcell",
            "--compartment",
            "both",
            "--membrane-channels",
            "CD3,CD20",
            "--post-processing",
            "legacy",
            "--fill-gaps",
            "a_full.tiff",
        ]
    )
    assert args.image_mpp == 1.0
    assert args.membrane_channels == "CD3,CD20"
    assert args.postprocessing_mode == "legacy"
    assert args.fill_gaps
