import sys
import types

import numpy as np
import pandas as pd
import pytest

from imc.segmentation import (
    MESMER_MAPS,
    align_cell_and_nuclear_masks,
    deepcell_postprocess_both_compartments,
    deepcell_segment,
    deepcell_segment_probabilities,
    mesmer_postprocess_kwargs,
    percentile_normalize,
    prepare_stack,
    qc_cell_and_nuclear_masks,
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
    assert args.membrane_mode == "max"
    assert args.whole_cell_threshold is None
    assert args.cell_expansion == 1.0
    assert args.min_cell_area == 15.0
    assert args.min_nucleus_area is None
    assert args.qc and not args.save_maps
    assert args.output_dir is None
    args = build_cli("segment").parse_args(
        ["--membrane-mode", "mean", "--no-qc", "--save-maps", "--output-dir", "out", "a.tiff"]
    )
    assert args.membrane_mode == "mean"
    assert not args.qc and args.save_maps
    assert args.output_dir.name == "out"


def test_membrane_image_max_mode():
    stack, _, _ = make_stack()
    kwargs = dict(membrane_channels=["CD45", "CD3"], normalization="percentile")
    cyto = prepare_stack(stack, LABELS, "cytoplasm", membrane_mode="max", **kwargs)
    expected = np.maximum(percentile_normalize(stack[3]), percentile_normalize(stack[4]))
    np.testing.assert_allclose(cyto, expected)
    assert (cyto >= prepare_stack(stack, LABELS, "cytoplasm", **kwargs) - 1e-12).all()


def test_mesmer_sizes_given_in_um2():
    cell, nuc = mesmer_postprocess_kwargs(whole_cell_threshold=0.05, min_cell_area=15)
    assert cell == dict(
        interior_threshold=0.05, small_objects_threshold=60, fill_holes_threshold=60
    )
    assert nuc == dict()
    cell, nuc = mesmer_postprocess_kwargs(min_cell_area=0, min_nucleus_area=4)
    assert cell == dict()
    assert nuc == dict(small_objects_threshold=16, fill_holes_threshold=16)
    cell, _ = mesmer_postprocess_kwargs(cell_expansion=1.0)
    assert cell == dict(pixel_expansion=2)


@pytest.fixture
def mesmer_calls(monkeypatch):
    """Fake Mesmer recording its settings, with per-pixel outputs at its own scale."""
    calls = list()

    class FakeMesmer:
        def _postprocess(self, output, **kwargs):
            shape = output["whole-cell"][0].shape[1:3]
            return np.ones((1,) + shape + (2,), dtype=np.int32)

        def predict(self, image, image_mpp=None, compartment="whole-cell", **kwargs):
            calls.append(dict(image_mpp=image_mpp, **kwargs))
            f = int(round(image_mpp / 0.5)) if image_mpp else 1
            shape = (1, image.shape[1] * f, image.shape[2] * f, 1)
            output = {
                "whole-cell": [np.full(shape, 0.1), np.full(shape, 0.2)],
                "nuclear": [np.full(shape, 0.3), np.full(shape, 0.4)],
            }
            return self._postprocess(output)[:, ::f, ::f]

    applications = types.ModuleType("deepcell.applications")
    applications.MultiplexSegmentation = FakeMesmer
    applications.NuclearSegmentation = FakeMesmer
    applications.CytoplasmSegmentation = FakeMesmer
    deepcell = types.ModuleType("deepcell")
    deepcell.applications = applications
    monkeypatch.setitem(sys.modules, "deepcell", deepcell)
    monkeypatch.setitem(sys.modules, "deepcell.applications", applications)
    return calls


def test_deepcell_gets_mesmer_settings(mesmer_calls):
    cell, nuc = mesmer_postprocess_kwargs(0.05, 15, None)
    image = to_deepcell_input(np.random.rand(2, 30, 40))
    deepcell_segment(
        image,
        "both",
        image_mpp=1.0,
        postprocess_kwargs_whole_cell=cell,
        postprocess_kwargs_nuclear=nuc,
    )
    assert mesmer_calls[0]["postprocess_kwargs_whole_cell"] == cell
    assert "postprocess_kwargs_nuclear" not in mesmer_calls[0]


def test_mesmer_maps_are_returned(mesmer_calls):
    image = to_deepcell_input(np.random.rand(2, 30, 40))
    pred, maps = deepcell_segment(image, "both", image_mpp=1.0, return_maps=True)
    assert pred.shape == (30, 40, 2)
    assert maps.shape == (len(MESMER_MAPS), 30, 40)
    np.testing.assert_allclose(maps[:, 10, 10], [0.2, 0.1, 0.4, 0.3], atol=1e-6)


class FakeROIWithFiles(FakeROI):
    def __init__(self, stack, labels, folder):
        super().__init__(stack, labels)
        self.folder = folder

    def get_input_filename(self, name):
        from imc.types import Path

        files = dict(
            stack="x_full.tiff", cell_mask="x_full_mask.tiff", nuclei_mask="x_full_nucmask.tiff"
        )
        return Path(self.folder) / files[name]


def test_segment_roi_writes_outputs_to_folder(mesmer_calls, tmp_path):
    stack, _, _ = make_stack()
    raw = tmp_path / "raw"
    raw.mkdir()
    out = segment_roi(
        FakeROIWithFiles(stack, LABELS, raw),
        model="deepcell",
        compartment="both",
        membrane_channels=["CD45", "CD3"],
        plot_segmentation=False,
        save_maps=True,
        output_dir=tmp_path / "new",
    )
    kwargs = mesmer_calls[0]["postprocess_kwargs_whole_cell"]
    assert "interior_threshold" not in kwargs
    assert kwargs["pixel_expansion"] == 2
    assert kwargs["small_objects_threshold"] == 60
    assert sorted(p.name for p in (tmp_path / "new").iterdir()) == [
        "x_full_cell_qc.csv",
        "x_full_mask.tiff",
        "x_full_mesmer_input.tiff",
        "x_full_mesmer_maps.tiff",
        "x_full_nucmask.tiff",
    ]
    assert list(raw.iterdir()) == []
    assert set(out) == {"cell", "nuclei", "cell_qc", "mesmer_maps"}


# Quality control after pairing
def scene_with_split_nucleus(dip=False):
    """Three whole nuclei, and one nucleus cut in two by a cell border."""
    shape = (40, 100)
    cell = np.zeros(shape, int)
    nuc = np.zeros(shape, int)
    cell[5:35, 5:20] = 1
    cell[5:35, 20:35] = 2
    nuc[disk(shape, (20, 20), 7)] = 1
    nuc[(nuc == 1) & (cell == 2)] = 2
    for i, x in enumerate([50, 70, 90], 3):
        cell[5:35, x - 9 : x + 9] = i
        nuc[disk(shape, (20, x), 7)] = i
    dna = (nuc > 0) * 100.0
    if dip:
        dna[:, 19:21] = 20
    return np.stack([cell, nuc], -1), dna


def test_nucleus_split_in_two_is_merged():
    mask, dna = scene_with_split_nucleus()
    out, table = qc_cell_and_nuclear_masks(mask, dna)
    check_pairing(out)
    assert out[..., 0].max() == 4
    assert out[20, 15, 0] == out[20, 25, 0]
    assert table["merged_pieces"].tolist() == [2, 1, 1, 1]
    assert (out[..., 1] > 0).sum() == (mask[..., 1] > 0).sum()


def test_nuclei_with_a_dna_dip_are_not_merged():
    mask, dna = scene_with_split_nucleus(dip=True)
    out, table = qc_cell_and_nuclear_masks(mask, dna)
    check_pairing(out)
    assert out[..., 0].max() == 5
    assert (table["merged_pieces"] == 1).all()


def test_small_cells_removed_and_partial_nuclei_flagged():
    cell = np.zeros(SHAPE, int)
    nuc = np.zeros(SHAPE, int)
    cell[5:25, 5:25] = 1
    nuc[disk(SHAPE, (15, 15), 6)] = 1
    cell[5:25, 30:50] = 2
    nuc[14:17, 39:42] = 2
    cell[30:33, 30:33] = 3
    cell[28:38, 5:20] = 4
    mask = np.stack([cell, nuc], -1)
    out, table = qc_cell_and_nuclear_masks(mask, np.zeros(SHAPE), min_cell_area=15)
    check_pairing(out)
    assert out[..., 0].max() == 3
    assert table["has_nucleus"].tolist() == [True, True, False]
    assert table["partial_nucleus"].tolist() == [False, True, False]


def test_empty_cells_without_nucleus_are_removed():
    cell = np.zeros(SHAPE, int)
    nuc = np.zeros(SHAPE, int)
    signal = np.zeros(SHAPE)
    for i, x in enumerate([5, 17, 29, 41], 1):
        cell[5:15, x : x + 10] = i
        nuc[8:12, x + 3 : x + 7] = i
    signal[5:15, 5:51] = 10
    cell[25:35, 5:15] = 5  # no nucleus, with marker signal
    signal[25:35, 5:15] = 10
    cell[25:35, 30:40] = 6  # no nucleus, over an empty area
    mask = np.stack([cell, nuc], -1)
    out, table = qc_cell_and_nuclear_masks(mask, np.zeros(SHAPE), signal_image=signal)
    check_pairing(out)
    assert out[..., 0].max() == 5
    assert out[30, 35, 0] == 0 and out[30, 10, 0] > 0
    assert (~table["has_nucleus"]).sum() == 1
