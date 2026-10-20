"""
Segmentation of image stacks using pretrained deep learning models
such as Stardist and DeepCell.
"""


import typing as tp
from functools import lru_cache, partial

import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
from skimage.exposure import equalize_hist as eq
from skimage.transform import resize
import tifffile
import sklearn

from imc.types import Array, Figure, Path, Series
from imc.data_models import roi as _roi
from imc.graphics import random_label_cmap
from imc.utils import minmax_scale


def normalize(image: Array, mode: str = "cyx") -> Array:
    if len(image.shape) == 2:
        return minmax_scale(eq(image))
    elif len(image.shape) == 3:
        if mode == "cyx":
            return np.asarray([minmax_scale(eq(x)) for x in image])
        elif mode == "yxc":
            return np.stack(
                [minmax_scale(eq(image[..., i])) for i in range(image.shape[-1])],
                axis=-1,
            )
    raise ValueError("Unknown image shape.")


def percentile_normalize(image: Array, lower: float = 1.0, upper: float = 99.8) -> Array:
    """Scale each channel of a YX or CYX image to [0, 1] between two percentiles."""
    if len(image.shape) == 2:
        lo, hi = np.percentile(image, [lower, upper])
        return np.clip((image - lo) / max(hi - lo, 1e-9), 0, 1)
    if len(image.shape) == 3:
        return np.asarray([percentile_normalize(x, lower, upper) for x in image])
    raise ValueError("Unknown image shape.")


def marker_names(channel_labels: Series) -> Series:
    """Marker part of channel labels, e.g. 'CD20(Dy161)' -> 'CD20'."""
    return channel_labels.str.replace(r"\(.*$", "", regex=True).str.strip()


# Resolution (um/px) at which DeepCell's Mesmer model works; images are rescaled to it
# and Mesmer counts object sizes in pixels at this scale.
MESMER_MPP = 0.5
# Per-pixel outputs of Mesmer used by its watershed, as returned by ``deepcell_segment``.
MESMER_MAPS = [
    "cell_interior",
    "cell_inner_distance",
    "nucleus_interior",
    "nucleus_inner_distance",
]


def nuclear_channels(channel_labels: Series) -> Array:
    """Boolean array marking the DNA (iridium) channels."""
    nuclear_chs = (
        channel_labels.str.contains(r"DNA|Iridium", regex=True)
        | channel_labels.str.contains("191Ir(Ir191)", regex=False)
        | channel_labels.str.contains("193Ir(Ir193)", regex=False)
    ).values
    if nuclear_chs.sum() == 0:
        raise ValueError(
            "Could not determine nuclear channels using 'DNA', 'Iridium', or Ir91/Ir193."
        )
    return nuclear_chs


def mesmer_postprocess_kwargs(
    whole_cell_threshold: float = None,
    min_cell_area: float = None,
    min_nucleus_area: float = None,
) -> tp.Tuple[tp.Dict[str, tp.Any], tp.Dict[str, tp.Any]]:
    """
    Mesmer post-processing settings for the whole-cell and nuclear compartments.

    Mesmer rescales images to ``MESMER_MPP`` before segmentation and removes objects,
    and fills holes, smaller than a number of pixels counted at that scale. Sizes are
    given here in square micrometres and converted to that scale. ``None`` or 0 keeps
    Mesmer's default of 15 pixels (about 4 um2).

    Parameters
    ----------
    whole_cell_threshold: float
        Threshold on the predicted cell interior probability, which sets how far cell
        outlines extend (Mesmer's ``interior_threshold``, by default 0.2).
    min_cell_area: float
        Smallest cell kept, in um2.
    min_nucleus_area: float
        Smallest nucleus kept, in um2.

    Returns
    -------
    Settings for ``postprocess_kwargs_whole_cell`` and ``postprocess_kwargs_nuclear``.
    """

    def size_kwargs(area: tp.Optional[float]) -> tp.Dict[str, tp.Any]:
        if not area:
            return dict()
        n = max(1, int(round(area / MESMER_MPP**2)))
        return dict(small_objects_threshold=n, fill_holes_threshold=n)

    cell = size_kwargs(min_cell_area)
    if whole_cell_threshold is not None:
        cell["interior_threshold"] = whole_cell_threshold
    return cell, size_kwargs(min_nucleus_area)


def prepare_stack(
    stack: Array,
    channel_labels: Series,
    compartment: str = "nuclear",
    channel_exclude: Series = None,
    membrane_channels: tp.Sequence[str] = None,
    normalization: str = "equalize",
    membrane_mode: str = "mean",
) -> Array:
    """
    Make the nuclear and/or membrane image used as segmentation input.

    Parameters
    ----------
    stack: Array
        Channel stack of shape (C, Y, X).
    channel_labels: Series
        Labels of the channels in ``stack``, e.g. 'CD20(Dy161)'.
    compartment: str
        One of ``nuclear``, ``cytoplasm`` or ``both``.
    channel_exclude: Series
        Boolean Series marking channels to exclude.
    membrane_channels: list[str]
        Markers averaged into the membrane (cytoplasm) image, matched to the part
        of the channel label before the metal tag. By default, all channels with an
        antibody are used except the nuclear channels, Ki67 and SMA.
    normalization: str
        ``equalize`` (histogram equalization, the input of the ilastik model) or
        ``percentile`` (scaling between the 1st and 99.8th percentiles).
    membrane_mode: str
        How the normalized membrane channels are combined: ``mean``, or ``max`` (the
        brightest marker in each pixel, so that a cell expressing only one of the
        markers is outlined as clearly as one expressing several).
    """
    assert compartment in ["nuclear", "cytoplasm", "both"]
    assert normalization in ["equalize", "percentile"]
    assert membrane_mode in ["mean", "max"]
    norm = normalize if normalization == "equalize" else percentile_normalize

    if channel_exclude is not None:
        keep = ~np.asarray(channel_exclude, dtype=bool)
        stack = stack[keep]
        channel_labels = channel_labels[keep]

    # Get nuclear channels
    nuclear_chs = nuclear_channels(channel_labels)
    nucl = norm(stack[nuclear_chs]).mean(0)
    if compartment == "nuclear":
        return nucl

    # Get membrane/cytoplasm channels
    if membrane_channels:
        names = marker_names(channel_labels)
        missing = [m for m in membrane_channels if m not in names.values]
        if missing:
            print(f"Membrane channels not found in stack and skipped: {missing}")
        cyto_chs = names.isin(list(membrane_channels)).values
        if cyto_chs.sum() == 0:
            raise ValueError("None of the membrane channels were found in the stack.")
    else:
        # channels without an antibody are labeled by mass only, e.g. '80ArAr(ArAr80)'
        no_antibody = channel_labels.str.match(r"^\d").values
        other = channel_labels.str.contains(r"Ki67|SMA", regex=True).values
        cyto_chs = ~(nuclear_chs | no_antibody | other)
    cyto = norm(stack[cyto_chs])
    cyto = cyto.max(0) if membrane_mode == "max" else cyto.mean(0)
    if compartment == "cytoplasm":
        return cyto

    # Combined together and expand to 4D
    return np.stack((nucl, cyto))


def extract_features(
    arr: tp.Union[Array, tp.Sequence[Array]],
    model: str = "VGG19",
    pyramids: tp.Sequence[int] = (1, 2, 4),
) -> tp.List[Array]:
    """
    Extract features from images using pre-trained deep-learning model
    """
    import tensorflow as tf

    assert model in ["VGG16", "VGG19"]

    if isinstance(arr, np.ndarray):
        assert len(arr[0].shape) == 3
        arr = [arr]
    elif isinstance(arr, (list, tuple)):
        for a in arr:
            assert len(a.shape) == 3, "Input must be of shape YXC or list of YXC arrays!"
    channels = np.unique([a.shape[-1] for a in arr])
    assert len(channels) == 1, "All arrays must have the same number of channels!"
    n_channels = channels[0]
    if n_channels == 1:
        print("Array contained only one channel. Repeating to RGB channels.")
        arr = [np.concatenate([a, a, a], axis=2).squeeze() for a in arr]
    elif n_channels != 3:
        print("Array did not contain 3 channels. Using mean of all channels.")
        arr = [a.mean(-1) for a in arr]
        arr = [np.asarray([a, a, a]).squeeze() for a in arr]

    shapes = {
        j: [(arr[j].shape[0] // i, (arr[j].shape[1] // i)) for i in pyramids]
        for j in range(len(arr))
    }
    unique_shapes = tuple(
        map(tuple, np.unique(np.concatenate(list(shapes.values())), axis=0))
    )

    models = dict()
    for shape in unique_shapes:
        f = getattr(getattr(tf.keras.applications, model.lower()), model)
        m = f(weights="imagenet", include_top=False, input_shape=(shape[0], shape[1], 3))
        for layer in m.layers:
            layer.trainable = False
        models[shape] = tf.keras.models.Model(inputs=m.input, outputs=m.layers[2].output)

    features = list()
    for img_idx, shpx in shapes.items():
        orig_shape = arr[img_idx].shape[:-1]
        _feat = list()
        for sh in shpx:
            feat = (
                models[sh]
                .predict(resize(arr[img_idx], sh + (3,))[np.newaxis, ...])
                .squeeze()
            )
            _feat.append(resize(feat, orig_shape + (feat.shape[-1],)))
        features.append(np.concatenate(np.asarray(_feat), axis=-1))
    return features


def train_pixel_classifier(
    x: Array,
    y: Array,
    output_file: Path = None,
    save_model: bool = True,
    model: str = "RandomForestClassifier",
    subsample_frac: float = None,
    subsample_n: float = None,
) -> sklearn.base.BaseEstimator:
    """
    Train classifier to predict pixel classes from image stacks (x) and sparse labels (y).

    Parameters
    ----------
    x: np.ndarray, list
        Array with shape BCXY (smaple,channel,X,Y) of signal
    y: np.ndarray, list
        Array with shape BXY (smaple,X,Y) of cellular compartments (nucleus, cytoplasm, background).
    output_file: pathlib.Path
        File to save model to. Traditionally ends in '.sav'.
        Only required if save_model is True.
    save_model: bool
        Whether to save the trained model.
    model: str
        Type of model to use. Only 'RandomForestClassifier' available for the time being.
    subsample_frac: float
        Fraction of pixels to train on. Defaults to all.
    subsample_n: float
        Number of pixels to train on. Defaults to all.
    """
    from sklearn.ensemble import RandomForestClassifier
    import joblib

    assert model == "RandomForestClassifier"

    if save_model:
        assert output_file is not None

    if subsample_frac is not None:
        assert subsample_n is None

    if subsample_n is not None:
        assert subsample_frac is None

    # TODO: make sure unequally shaped samples are supported
    print("Extracting features.")
    xf = np.asarray(extract_features([xx for xx in x]))
    xf = xf.reshape((-1, xf.shape[-1]))
    yf = y.reshape((xf.shape[0],))

    # remove null pixels
    nan = pd.isnull(xf).any(1)
    xf = xf[~nan, :]
    yf = yf[~nan]

    # select only pixels with class
    sel = yf > 0
    xf = xf[sel, :]
    yf = yf[sel]

    # subsample if requested
    if subsample_frac is not None:
        subsample_n = int(yf.shape[0] * subsample_frac)

    if subsample_n is not None:
        c = np.random.choice(range(yf.shape[0]), subsample_n)
        xf = xf[c, :]
        yf = yf[c]

    print("Training model.")
    _model = RandomForestClassifier(n_jobs=-1)
    _model.fit(xf, yf)

    if save_model:
        print("Saving model.")
        joblib.dump(_model, output_file)
    return _model


def predict_pixel_classifier(x: Array, model_file: Path) -> Array:
    """
    Predict pixel classes from image stacks (x) using trained model.

    Parameters
    ----------
    x: np.ndarray, list
        Array with shape BCXY (smaple,channel,X,Y) of signal
    model_file: pathlib.Path
        File with trained model to load.
    """
    import joblib

    print("Loading model.")
    _model = joblib.load(model_file)

    print("Extracting features.")
    xf = np.asarray(extract_features([xx for xx in x]))
    shape = xf.shape[:-1] + (3,)
    xf = xf.reshape((-1, xf.shape[-1]))
    # fill null pixels with 0
    xf[pd.isnull(xf)] = 0
    print("Predicting.")
    probs = _model.predict_proba(xf)
    probs = probs.reshape(shape)
    return probs


def stardist_segment_nuclei(image: Array, model_str: str = "2D_versatile_fluo") -> Array:
    from stardist.models import StarDist2D

    model = StarDist2D.from_pretrained(model_str)
    mask, _ = model.predict_instances(image)
    return mask.astype("uint")


def deepcell_resize_to_predict_shape(image: Array) -> Array:
    """Resize an image to 256x256. Not used: DeepCell tiles images of any size."""
    if len(image.shape) == 2:
        return resize(image, (256, 256))[np.newaxis, :, :, np.newaxis]
    if len(image.shape) == 3:
        return resize(np.moveaxis(image, 0, -1), (256, 256, image.shape[0]))[
            np.newaxis, ...
        ]
    raise ValueError("Image with unknown shape.")


def to_deepcell_input(image: Array) -> Array:
    """Reshape a YX or CYX image to the BYXC shape expected by DeepCell."""
    if len(image.shape) == 2:
        return image[np.newaxis, :, :, np.newaxis]
    if len(image.shape) == 3:
        return np.moveaxis(image, 0, -1)[np.newaxis, ...]
    raise ValueError("Image with unknown shape.")


def resize_labels(mask: Array, shape: tp.Tuple[int, ...]) -> Array:
    """Resize a label mask with nearest-neighbour interpolation, keeping label values."""
    if tuple(mask.shape) == tuple(shape):
        return mask
    out = resize(
        mask, shape, order=0, preserve_range=True, anti_aliasing=False, mode="edge"
    )
    return out.astype(mask.dtype)


def split_image_into_tiles(
    image: Array,
    tile_shape: tp.Tuple[int, int] = (256, 256),
    overlap: tp.Tuple[int, int] = (0, 0),
    pad_to_tile_shape: bool = False,
) -> tp.Dict[tp.Tuple[int, int], Array]:
    shape = np.asarray(image.shape)
    x, y = np.asarray(tile_shape)

    r1, r2 = shape / tile_shape
    n, m = shape // tile_shape + 2
    n += int(r1)
    m += int(r2)

    tiles = dict()
    for i in range(m):
        for j in range(n):
            offsets = overlap[0] * i, overlap[1] * j
            start = (max(i * x - offsets[0], 0), max(j * y - offsets[1], 0))
            end = (
                ((i + 1) * x - offsets[0]) if (i + 2) < n else shape[0] - start[0],
                ((j + 1) * y - offsets[1]) if (j + 2) < m else shape[1] - start[1],
            )
            tile = image[start[0] : end[0], start[1] : end[1]]
            if pad_to_tile_shape:
                diff = np.asarray(tile.shape) - tile_shape
                if not (diff > 0).all():
                    tile = np.pad(tile, ((0, abs(diff[0])), (0, abs(diff[1]))))
            if (min(tile.shape) > 0) and tile.sum() > 0:
                tiles[start] = tile

    return tiles


def join_tiles_to_image(
    tiles: tp.Dict[tp.Tuple[int, int], Array], trim_all_zeros: bool = True
) -> Array:
    grid = np.asarray(list(tiles.keys()))

    image_shape = np.array([0, 0])
    first_row_value = grid[0][0]
    for pos, tile in tiles.items():
        if pos[0] != first_row_value:
            continue
        image_shape[1] += tile.shape[0]
    first_col_value = grid[0][1]
    for pos, tile in tiles.items():
        if pos[1] != first_col_value:
            continue
        image_shape[0] += tile.shape[1]

    rec = np.zeros(image_shape, dtype=tile.dtype)
    for (i, j), arr in tiles.items():
        x, y = arr.shape
        rec[i : i + x, j : j + y] = arr
        rec[i : i + x, j : j + y] = arr

    # clip
    if trim_all_zeros:
        rec = rec[:, rec.sum(0) > 0]
        rec = rec[rec.sum(1) > 0, :]

    return rec


@lru_cache(maxsize=None)
def _deepcell_application(app_class: tp.Any) -> tp.Any:
    """One instance per DeepCell application, so that its model is loaded only once."""
    return app_class()


def _predict_with_maps(
    app: tp.Any, image: Array, image_mpp: float, kwargs: tp.Dict[str, tp.Any]
) -> tp.Tuple[Array, Array]:
    """Run Mesmer and also return the per-pixel outputs its watershed starts from."""
    from skimage.transform import resize as resize_image

    captured = dict()
    postprocess = app._postprocess

    def keep_output(output, **postprocess_kwargs):
        captured["output"] = output
        return postprocess(output, **postprocess_kwargs)

    app._postprocess = keep_output
    try:
        pred = app.predict(image, image_mpp=image_mpp, **kwargs).squeeze()
    finally:
        app._postprocess = postprocess
    out = captured["output"]
    # Mesmer gives [inner distance, interior probability] for each compartment
    layers = [out["whole-cell"][1], out["whole-cell"][0], out["nuclear"][1], out["nuclear"][0]]
    shape = image.shape[1:3]
    maps = np.stack(
        [
            resize_image(
                np.squeeze(np.asarray(x)).astype(np.float32),
                shape,
                order=1,
                anti_aliasing=True,
                preserve_range=True,
            )
            for x in layers
        ]
    )
    return pred, maps.astype(np.float32)


def deepcell_segment(
    image: Array,
    compartment: str = None,
    image_mpp: float = None,
    use_tilling: bool = False,
    postprocess_kwargs_whole_cell: tp.Dict[str, tp.Any] = None,
    postprocess_kwargs_nuclear: tp.Dict[str, tp.Any] = None,
    return_maps: bool = False,
) -> tp.Union[Array, tp.Tuple[Array, Array]]:
    """
    Segment an image with a DeepCell model.

    Parameters
    ----------
    image: Array
        Image of shape (1, Y, X, C), see ``to_deepcell_input``.
    compartment: str
        One of ``nuclear``, ``cytoplasm`` or ``both`` (Mesmer).
    image_mpp: float
        Pixel size of ``image`` in micrometres.
    postprocess_kwargs_whole_cell, postprocess_kwargs_nuclear: dict
        Post-processing settings passed to Mesmer (compartment ``both``), for example
        from ``mesmer_postprocess_kwargs``.
    return_maps: bool
        Whether to also return Mesmer's per-pixel outputs (compartment ``both``, without
        tiling), an array of shape (4, Y, X) with the layers in ``MESMER_MAPS``.
    """
    from deepcell.applications import (
        MultiplexSegmentation,
        NuclearSegmentation,
        CytoplasmSegmentation,
    )

    if compartment is None:
        # If image has more than one channel i.e. CYX
        # assume segmentation of both compartments, otherwise nuclear
        compartment = "both" if image.shape[-1] > 1 else "nuclear"

    kwargs = dict()
    if compartment == "nuclear":
        app = _deepcell_application(NuclearSegmentation)
    if compartment == "cytoplasm":
        app = _deepcell_application(CytoplasmSegmentation)
    if compartment == "both":
        app = _deepcell_application(MultiplexSegmentation)
        kwargs = dict(compartment=compartment)
        if postprocess_kwargs_whole_cell:
            kwargs["postprocess_kwargs_whole_cell"] = dict(postprocess_kwargs_whole_cell)
        if postprocess_kwargs_nuclear:
            kwargs["postprocess_kwargs_nuclear"] = dict(postprocess_kwargs_nuclear)

    maps = None
    if return_maps:
        if compartment != "both" or use_tilling:
            raise ValueError("Maps are only returned for compartment 'both' without tiling.")
        pred, maps = _predict_with_maps(app, image, image_mpp, kwargs)
    elif not use_tilling:
        pred = app.predict(image, image_mpp=image_mpp, **kwargs).squeeze()
    else:
        tiles = split_image_into_tiles(image.squeeze())

        # together as batches
        a = np.asarray(list(tiles.values()))[..., np.newaxis]
        preds = app.predict(a, image_mpp=image_mpp, **kwargs).squeeze()
        pred = join_tiles_to_image(dict(zip(tiles.keys(), preds)))

        # # one by one
        # preds = dict()
        # for pos, tile in tiles.items():
        #     print([pos])
        #     if tile.sum() == 0:
        #         preds[pos] = tile
        #         continue
        #     preds[pos] = app.predict(
        #         tile[np.newaxis, ..., np.newaxis], image_mpp=1
        #     ).squeeze()
        # pred = join_tiles_to_image(preds)

    if pred.shape[:2] != image.shape[1:3]:
        pred = resize_labels(pred, image.shape[1:3] + pred.shape[2:])
    if return_maps:
        return pred, maps
    return pred


def cellpose_segment(
    image: Array, compartment: tp.Union[tp.Literal["nuclear"], tp.Literal["cytoplasm"]]
) -> Array:
    from cellpose import models

    assert compartment in ["nuclear", "cytoplasm"]

    comp = {"nuclear": "nuclei", "cytoplasm": "cyto"}
    if compartment == "nuclear":
        channels = [0, 0]
    elif compartment == "cytoplasm":
        raise NotImplementedError
        channels = [1, 2]
    model = models.Cellpose(gpu=False, model_type=comp[compartment])
    masks, flows, styles, diams = model.eval(
        [image],
        normalize=False,
        diameter=None,
        channels=channels,
    )
    return masks[0]


def plot_cellpose_output(
    image: Array, masks: tp.Sequence[Array], flows: tp.Sequence[Array]
) -> Figure:
    n = len(masks)

    fig, axes = plt.subplots(
        n, 5, figsize=(5 * 4, n * 4), sharex=True, sharey=True, squeeze=False
    )
    for i in range(n):
        axes[i][0].imshow(image)
        m = np.ma.masked_array(masks[i], masks[i] == 0)
        axes[i][1].imshow(m)
        f = resize(flows[i][2], image.shape)
        axes[i][2].imshow(f)
        f = resize(flows[i][0], image.shape)
        axes[i][3].imshow(f)
        f = resize(flows[i][0].mean(-1), image.shape)
        axes[i][4].imshow(f)
    for ax in axes.flat:
        ax.axis("off")
    labs = [
        "Original image",
        "Predicted mask",
        "Flows1",
        "Flows2",
        "mean(Flows2)",
    ]
    for ax, lab in zip(axes[0], labs):
        ax.set(title=lab)
    return fig


def cellpose_postprocessing(image: Array, mask: Array, flow: Array):
    from skimage import filters

    flo = flow[0].mean(-1)
    flo = flo / flo.max()

    image2 = resize(image, flo.shape)
    mask2 = resize(mask > 0, flo.shape)
    mask2 = np.ma.masked_array(mask2, mask2 == 0)

    algos = ["li", "mean", "minimum", "otsu", "triangle", "yen", "isodata"]

    segs = dict()
    _perf = dict()
    fig, axes = plt.subplots(3, 5, sharex=True, sharey=True)
    axes[0][0].imshow(image2)
    axes[1][0].imshow(mask2)
    axes[2][0].imshow(flo)  # np.ma.masked_array(mask, mask==0)
    for algo, ax in zip(algos, axes[:, 1:].flat):
        f = getattr(filters, f"threshold_{algo}")
        t = f(flo)
        segs[algo] = flo > t
        s = flo[segs[algo]].sum()
        r = s / segs[algo].sum()
        n = ndi.label(segs[algo])[1]
        _perf[algo] = (s, r, n)
        ax.imshow(segs[algo])
        ax.set(title=f"{algo}:\nsum = {s:.1f}, ratio = {r:.2f}, n = {n}")

    perf = pd.DataFrame(_perf, index=["sum", "ratio", "objs"]).T
    perf["objs"] = 5 + (2 ** np.log1p(perf["objs"]))
    perf["sum_norm"] = perf["sum"] / perf["sum"].sum()
    perf["ratio_norm"] = perf["ratio"] / perf["ratio"].sum()
    perf["weight"] = perf[["sum_norm", "ratio_norm"]].mean(1)

    seg = np.asarray(list(segs.values()))
    seg_t = np.average(seg, axis=0, weights=perf["weight"])  # > 0.5
    axes[-1][-2].imshow(seg_t)
    axes[-1][-2].set(title="Mean of thresholding algorightms")
    axes[-1][-1].imshow(seg_t > 0.5)
    axes[-1][-1].set(title="Threshold of mean")
    for ax in axes.flat:
        ax.axis("off")

    fig, ax = plt.subplots()
    ax.scatter(*perf.T.values)
    for algo in perf.index:
        ax.text(perf.loc[algo, "sum"], perf.loc[algo, "ratio"], s=algo)

    return seg_t > 0.5


def try_all_segmentations(
    image: Array,
    compartment: str = "nuclear",
    models: tp.Sequence[str] = ["stardist", "deepcell", "cellpose"],
) -> Array:
    """
    Try out various models for segmentation.
    Only nuclear available so far.
    """
    assert compartment in ["nuclear"]

    masks = np.zeros((len(models), *image.shape))
    i = 0
    if "stardist" in models:
        masks[i] = stardist_segment_nuclei(image)
        i += 1

    if "deepcell" in models:
        masks[i] = deepcell_segment(
            image[np.newaxis, ..., np.newaxis], compartment=compartment
        )
        i += 1

    if "cellpose" in models:
        masks[i] = cellpose_segment(image, compartment=compartment)
        i += 1

    return masks


def plot_all_segmentations(
    masks: Array, image: Array = None, model_names: tp.Sequence[str] = None
) -> Figure:
    """
    Plot the output of various segmentation models.
    """
    if image is None:
        image = np.zeros(masks.shape[1:])
    if model_names is None:
        model_names = [f"Model {i}" for i, _ in enumerate(masks, 1)]

    fig = plot_image_and_mask(image=image, mask_dict=dict(zip(model_names, masks)))
    return fig


def inflection_point(curve: tp.Sequence[float]) -> int:
    """Return the index of the inflection point of a curve"""
    from numpy.matlib import repmat

    n_points = len(curve)
    all_coord = np.vstack((range(n_points), curve)).T
    line_vec = all_coord[-1] - all_coord[0]
    line_vec_norm = line_vec / np.sqrt(np.sum(line_vec**2))
    vec_from_first = all_coord - all_coord[0]
    scalar_product = np.sum(vec_from_first * repmat(line_vec_norm, n_points, 1), axis=1)
    vec_to_line = vec_from_first - np.outer(scalar_product, line_vec_norm)
    return np.argmax(np.sqrt(np.sum(vec_to_line**2, axis=1)))


def plot_image_and_mask(image: Array, mask_dict: tp.Dict[str, Array]) -> Figure:
    """

    image: np.ndarray

    mask_dict: dict[str, np.ndarray]
    """
    cols = 1 + len(mask_dict)

    # Make RGB if multi channel
    if len(image.shape) == 3:
        if image.shape[0] == 2:
            image = np.stack([image[0], image[1], np.zeros(image.shape[1:])])
        image = np.moveaxis(image, 0, -1)

    for comp in mask_dict:
        mask_dict[comp] = np.ma.masked_array(mask_dict[comp], mask_dict[comp] == 0)

    fig, axes = plt.subplots(1, cols, figsize=(4 * cols, 4), sharex=True, sharey=True)
    axes[0].imshow(image, rasterized=True)
    axes[0].set(title="Original signal")
    cmap = random_label_cmap()
    for comp, ax in zip(mask_dict, axes[1:]):
        ax.imshow(mask_dict[comp], cmap=cmap, interpolation="none")
        ax.set(title=f"{comp.capitalize()} mask")
    # for i, comp in enumerate(mask_dict):
    #     ax = axes[1 + len(mask_dict) + i]
    #     ax.imshow(image, rasterized=True)
    #     cs = ax.contour(mask_dict[comp], cmap="Reds")
    #     # important to rasterize contours
    #     for c in cs.collections:
    #         try:
    #             c.set_rasterized(True)
    #         except Exception:
    #             pass
    #     ax.set(title=f"{comp.capitalize()} mask")
    for ax in axes:
        ax.axis("off")
    return fig


def deepcell_segment_probabilities(
    probabilities: Array,
    compartment: str,
    image_mpp: float = 1.0,
    postprocess_kwargs_whole_cell: tp.Dict[str, tp.Any] = None,
    postprocess_kwargs_nuclear: tp.Dict[str, tp.Any] = None,
) -> Array:
    """
    Segment ilastik probabilities (YXC: nuclear, cytoplasm, background) with the
    DeepCell MultiplexSegmentation (Mesmer) model.

    Parameters
    ----------
    probabilities: Array
        Probability array of shape (Y, X, 3).
    compartment: str
        One of ``cell``, ``cytoplasm``, ``nuclear`` or ``both``.
    image_mpp: float
        Pixel size of ``probabilities`` in micrometres. Mesmer was trained at
        0.5 um/px and rescales the input by ``image_mpp / 0.5``.
    postprocess_kwargs_whole_cell, postprocess_kwargs_nuclear: dict
        Post-processing settings passed to Mesmer, for example from
        ``mesmer_postprocess_kwargs``.
    """
    from deepcell.applications import MultiplexSegmentation

    # translate compartments
    imc2dc = {
        "cell": "whole-cell",
        "cytoplasm": "whole-cell",
        "nuclear": "nuclear",
        "both": "both",
    }

    kwargs = dict()
    if postprocess_kwargs_whole_cell:
        kwargs["postprocess_kwargs_whole_cell"] = dict(postprocess_kwargs_whole_cell)
    if postprocess_kwargs_nuclear:
        kwargs["postprocess_kwargs_nuclear"] = dict(postprocess_kwargs_nuclear)
    app = _deepcell_application(MultiplexSegmentation)
    pred: Array = app.predict(
        probabilities[np.newaxis, ..., :-1],
        image_mpp=image_mpp,
        compartment=imc2dc[compartment],
        **kwargs,
    )
    return pred.squeeze()


def deepcell_postprocess_both_compartments(
    mask: Array,
    dominant_nuclei_threshold: float = 0.8,
    plot: bool = False,
    roi: _roi.ROI = None,
    fig_output_prefix: Path = None,
    verbose: bool = False,
) -> Array:
    """
    Function to perform post-processing of DeepCell masks when using
    the MultiplexSegmentation model with the 'both' compartment.

    The major goal is the alignment of the cellular and nuclear masks
    as much as possible. Cells overlapping several nuclei without a dominant one,
    and cells sharing a nucleus without a dominant cell, are removed; cells without
    a nucleus get the cell area minus its border as nucleus. Kept to reproduce
    earlier masks; ``align_cell_and_nuclear_masks`` keeps all objects.
    """
    import seaborn as sns
    from skimage.segmentation import find_boundaries

    def inspect_mask_pair(
        cell_mask: Array, nuclei_mask: Array
    ) -> tp.Tuple[int, list, dict]:
        match = 0
        unpaireds = list()
        multiplets = dict()
        for cell in np.unique(cell_mask)[1:]:
            o = np.unique(nuclei_mask[cell_mask == cell])
            o = [x for x in o if x != 0]
            if not o:
                # cells with no nucleus
                unpaireds.append(cell)
            elif len(o) == 1:
                # cells with exactly one nucleus
                match += 1
            else:
                # cells with more than one nucleus
                t = (cell_mask == cell).sum()
                fractions = dict()
                for nuclei in o:
                    m = (nuclei_mask == nuclei) & (cell_mask == cell)
                    fractions[nuclei] = len(cell_mask[m]) / t
                s = sum(fractions.values())
                multiplets[cell] = {k: v / s for k, v in fractions.items()}
        return match, unpaireds, multiplets

    def apply_postprocess(
        cell_mask: Array,
        nuclei_mask: Array,
        max_iter: int = 5,
        complement_singlets: bool = True,
        remove_border_from_complement: bool = False,
        remove_ambigous: bool = True,
    ) -> tp.Tuple[Array, Array]:
        it = 0
        match, unpaireds, multiplets = inspect_mask_pair(cell_mask, nuclei_mask)
        while not ((len(unpaireds) == 0) and (len(multiplets) == 0)):
            if it == max_iter:
                if verbose:
                    print("Reached maximum number of iterations.")
                break
            match, unpaireds, multiplets = inspect_mask_pair(cell_mask, nuclei_mask)
            # # 1. fix cells without nuclei
            if complement_singlets:
                # # # set nuclei to the whole mask minus one
                max_c = nuclei_mask.max()
                n_new = np.zeros(nuclei_mask.shape, dtype=int)
                for i, cell in enumerate(unpaireds, 1):
                    nuclei_mask[cell_mask == cell] = max_c + i
                    n_new[cell_mask == cell] = max_c + i
                # remove 1 pixel from border
                if remove_border_from_complement:
                    b = find_boundaries(n_new, 1, mode="inner", background=0)
                    nuclei_mask[b] = 0
                if verbose:
                    print(f"Added {len(unpaireds)} nuclei.")

            # # 2. fix cells overlaping more than one nuclei
            # # # If the matching is above threshold (i.e. 80%) then remove non-dominant nuclei
            # # # Else remove nuclei and cell
            if remove_ambigous:
                match, unpaireds, multiplets = inspect_mask_pair(cell_mask, nuclei_mask)
                n_removed = 0
                for cell, props in multiplets.items():
                    if max(props.values()) > dominant_nuclei_threshold:
                        dominant = pd.Series(props).idxmax()
                        nuclei_mask[
                            (cell_mask == cell)
                            & (nuclei_mask != dominant)
                            & (nuclei_mask > 0)
                        ] = dominant
                    else:
                        m = cell_mask == cell
                        cell_mask[cell_mask == cell] = 0
                        rem = np.unique(nuclei_mask[m])
                        nuclei_mask[np.isin(nuclei_mask, rem)] = 0
                        n_removed += 1
                if verbose:
                    print(f"Removed {n_removed} cells with ambiguous nuclei assignments.")
            it += 1
        return cell_mask, nuclei_mask

    if (mask.ndim != 3) or (mask.shape[-1] != 2):
        raise ValueError(
            "Array must contain more than one segmentation layer of shape "
            "(X,Y,2), the last dimension assumed to be cell and nuclei."
        )

    cell_mask, nuclei_mask = mask[..., 0].copy(), mask[..., 1].copy()

    # Inspect cell-nuclei relationship
    lc = len(np.unique(cell_mask)) - 1
    ln = len(np.unique(nuclei_mask)) - 1
    pre_match, pre_unpaireds, pre_multiplets = inspect_mask_pair(cell_mask, nuclei_mask)

    if verbose:
        print("Statistics prior postprocessing:")
        print(f"\tSegmentation has {lc} cells and {ln} nuclei.")
        print(f"\t{pre_match} exact matches between cells and nuclei.")
        print(f"\t{len(pre_unpaireds)} cells without nuclei.")
        print(f"\t{len(pre_multiplets)} cells overlaping more than one nuclei.")
        fracts = np.asarray([max(v.values()) for k, v in pre_multiplets.items()])
        print(
            "Found mean overlap between cells and nuclei in "
            f"multiplets to be {fracts.mean():.3f}(+/-){fracts.std():.3f}."
        )

    # Postprocess
    print("Applying postprocessing to segmentation masks...")
    cell_mask, nuclei_mask = apply_postprocess(
        cell_mask, nuclei_mask, remove_border_from_complement=True
    )
    nuclei_mask, cell_mask = apply_postprocess(
        nuclei_mask,
        cell_mask,
        complement_singlets=False,
        remove_ambigous=True,
        remove_border_from_complement=False,
    )
    cell_mask, nuclei_mask = apply_postprocess(
        cell_mask, nuclei_mask, remove_border_from_complement=True
    )

    # Report postprocessed stats:
    post_match, post_unpaireds, post_multiplets = inspect_mask_pair(
        cell_mask, nuclei_mask
    )
    if verbose:
        lc = len(np.unique(cell_mask)) - 1
        ln = len(np.unique(nuclei_mask)) - 1
        print("After postprocessing stats:")
        print(f"\tSegmentation has {lc} cells and {ln} nuclei.")
        print(f"\t{post_match} exact matches between cells and nuclei.")
        print(f"\t{len(post_unpaireds)} cells without nuclei.")
        print(f"\t{len(post_multiplets)} cells overlaping more than one nucleus.")

    # Report postprocessed stats:
    post_match, post_unpaireds, post_multiplets = inspect_mask_pair(
        nuclei_mask, cell_mask
    )
    if verbose:
        print(f"\t{post_match} exact matches between nuclei and cells.")
        print(f"\t{len(post_unpaireds)} nuclei without cells.")
        print(f"\t{len(post_multiplets)} nuclei overlaping more than one cell.")

    # # 3. align nuclei and cell integer IDs
    new_cell = np.zeros(cell_mask.shape, dtype="uint64")
    # could probably downcast these
    new_nuclei = np.zeros(cell_mask.shape, dtype="uint64")
    for i, cell in enumerate(np.unique(cell_mask)[1:], 1):
        new_cell[cell_mask == cell] = i
        # # find out number of nuclei mask:
        sel = np.unique(nuclei_mask[cell_mask == cell])
        try:
            old = [x for x in sel if x != 0][0]
            new_nuclei[nuclei_mask == old] = i
        except IndexError:
            pass
    cell_mask = new_cell
    nuclei_mask = new_nuclei

    if plot:
        reason = "If `plot`, `fig_output_prefix` must be given!"
        assert fig_output_prefix is not None, reason
        rows = 3 if roi is None else 4
        fig, axes = plt.subplots(
            rows, 2, figsize=(2 * 4, rows * 4), sharex=True, sharey=True
        )
        cmap = random_label_cmap()
        for i, (tmpmask, layer) in enumerate(
            [(mask[..., 0], "Cell mask"), (mask[..., 1], "Nuclei mask")]
        ):
            tmpmask = np.ma.masked_array(tmpmask, mask=tmpmask == 0)
            axes[0][i].imshow(tmpmask, cmap=cmap, interpolation="none")
            axes[0][i].set(title=f"Prior: {layer}")

        c_unpaired = mask[..., 0].copy()
        c_unpaired[~np.isin(mask[..., 0], np.unique(np.asarray(pre_unpaireds)))] = 0
        c_multiplets = mask[..., 0].copy()
        c_multiplets[~np.isin(mask[..., 0], np.unique(list(pre_multiplets.keys())))] = 0
        axes[1][0].imshow(c_multiplets, cmap=cmap, interpolation="none")
        axes[1][0].set(title="Cells overlapping multiple nuclei")
        axes[1][1].imshow(c_unpaired, cmap=cmap, interpolation="none")
        axes[1][1].set(title="Cells without nuclei")

        for i, (tmpmask, layer) in enumerate(
            [(cell_mask, "Cell mask"), (nuclei_mask, "Nuclei mask")]
        ):
            tmpmask = np.ma.masked_array(tmpmask, mask=tmpmask == 0)
            axes[2][i].imshow(tmpmask, cmap=cmap, interpolation="none")
            axes[2][i].set(title=f"Post: {layer}")

        if roi is not None:
            try:
                for ax, ch in zip(axes[3], ["DNA", "mean"]):
                    ax.imshow(
                        np.log1p(roi._get_channel(ch, dont_warn=True)[1]),
                        interpolation="bilinear",
                    )
                    ax.set(title=ch)
            except ValueError:
                pass
        for ax in axes.flat:
            ax.axis("off")
        fig.savefig(
            fig_output_prefix + "deepcell.cell-nuclei_relationships.svg",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

    # # Visualize fraction of cells with clear dominant vs ambiguous nuclei
    if plot:
        assert fig_output_prefix is not None, reason
        fig, ax = plt.subplots()
        sns.histplot(fracts, ax=ax)
        ax.axvline(dominant_nuclei_threshold, color="grey", linestyle="--")
        ax.set(
            title="Cells overlapping more than one nuclei",
            xlabel="Fraction of dominant nuclei in cell",
            ylabel="Cell count",
        )
        fig.savefig(
            fig_output_prefix
            + "deepcell.cell-nuclei.dominant_nuclei_area_distribution.svg",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

    return np.moveaxis(np.stack([cell_mask, nuclei_mask]), 0, -1)


def _label_overlaps(a: Array, b: Array) -> pd.DataFrame:
    """Number of pixels shared by each pair of overlapping labels in two masks."""
    sel = (a > 0) & (b > 0)
    nb = int(b.max()) + 1
    key = a[sel].astype(np.int64) * nb + b[sel].astype(np.int64)
    u, n = np.unique(key, return_counts=True)
    return pd.DataFrame({"a": u // nb, "b": u % nb, "n": n})


def _keep_largest_piece(labels: Array) -> Array:
    """Keep only the largest connected piece of each label."""
    from skimage.measure import label as label_pieces

    pieces = label_pieces(labels, background=0, connectivity=1)
    if pieces.max() == 0:
        return labels
    sel = pieces > 0
    df = pd.DataFrame({"piece": pieces[sel], "label": labels[sel]})
    df = df.drop_duplicates("piece")
    df["size"] = np.bincount(pieces.ravel())[df["piece"].values]
    largest = df.sort_values("size", ascending=False).drop_duplicates("label")
    keep = np.zeros(pieces.max() + 1, dtype=bool)
    keep[largest["piece"].values] = True
    out = labels.copy()
    out[~keep[pieces]] = 0
    return out


def align_cell_and_nuclear_masks(
    mask: Array,
    dominant_nuclei_threshold: float = 0.8,
    min_nucleus_area: int = 10,
    nucleus_expansion: int = 2,
    nuclear_image: Array = None,
    fill_gaps: bool = False,
    verbose: bool = False,
) -> Array:
    """
    Pair the cell and nuclear masks of the DeepCell MultiplexSegmentation model
    ('both' compartment) one to one without discarding objects.

    Rules, in order:
        1. A nucleus overlapping several cells is given whole to the cell holding
           more than ``dominant_nuclei_threshold`` of it; otherwise it is divided
           along the cell borders. Cells are extended to contain their nucleus.
        2. A nucleus outside all cells becomes a cell: the nucleus expanded by
           ``nucleus_expansion`` pixels into unassigned pixels.
        3. Nuclei within one cell are merged into the largest one if it holds more
           than ``dominant_nuclei_threshold`` of their area. Otherwise the cell is
           divided between its nuclei by seeded watershed. Nuclear fragments smaller
           than ``min_nucleus_area`` pixels in such cells are dropped.
        4. Cells without a nucleus are kept, with no nucleus (0 in the nuclear mask).
        5. If ``fill_gaps``, nuclei found in ``nuclear_image`` (DNA) in pixels not
           covered by any cell are added as cells, expanded as in rule 2.

    Nuclei take the label of their cell and labels are made consecutive.

    Parameters
    ----------
    mask: Array
        Array of shape (Y, X, 2) with the cell and nuclear masks.
    nuclear_image: Array
        DNA image of shape (Y, X); only used if ``fill_gaps``.

    Returns
    -------
    Array of shape (Y, X, 2) with the cell and nuclear masks.
    """
    from skimage.segmentation import watershed, expand_labels, relabel_sequential

    if (mask.ndim != 3) or (mask.shape[-1] != 2):
        raise ValueError("Mask must have shape (Y, X, 2) with cell and nuclear masks.")

    cell = relabel_sequential(mask[..., 0].astype(np.int64))[0]
    nuc = relabel_sequential(mask[..., 1].astype(np.int64))[0]
    stats = {"cells_in": int(cell.max()), "nuclei_in": int(nuc.max())}

    # 1. Nuclei overlapping several cells
    ov = _label_overlaps(nuc, cell).rename(columns={"a": "nucleus", "b": "cell"})
    ov["share"] = ov["n"] / ov.groupby("nucleus")["n"].transform("sum")
    top = ov.sort_values("share", ascending=False).drop_duplicates("nucleus")
    shared = ov.groupby("nucleus").size()
    shared = shared.index[shared > 1]
    dominant = top[top["share"] > dominant_nuclei_threshold]
    divided = top.loc[top["share"] <= dominant_nuclei_threshold, "nucleus"].values
    stats["nuclei_given_to_one_cell"] = int(dominant["nucleus"].isin(shared).sum())
    stats["nuclei_divided"] = len(divided)

    owner = np.zeros(nuc.max() + 1, dtype=np.int64)
    owner[dominant["nucleus"].values] = dominant["cell"].values
    sel = owner[nuc] > 0
    cell[sel] = owner[nuc[sel]]

    is_divided = np.zeros(nuc.max() + 1, dtype=bool)
    is_divided[divided] = True
    sel = is_divided[nuc]
    nuc[sel & (cell == 0)] = 0
    sel = sel & (cell > 0)
    if sel.any():
        key = nuc[sel] * (cell.max() + 1) + cell[sel]
        _, inv = np.unique(key, return_inverse=True)
        nuc[sel] = nuc.max() + 1 + inv.ravel()

    # 2. Nuclei outside all cells
    orphans = np.setdiff1d(np.arange(1, stats["nuclei_in"] + 1), ov["nucleus"].values)
    stats["nuclei_made_cells"] = len(orphans)
    if len(orphans):
        grown = expand_labels(np.where(np.isin(nuc, orphans), nuc, 0), nucleus_expansion)
        free = (grown > 0) & (cell == 0)
        cell[free] = grown[free] + cell.max()

    cell = _keep_largest_piece(cell)
    nuc[cell == 0] = 0

    # 3. Cells with several nuclei
    ov = _label_overlaps(cell, nuc).rename(columns={"a": "cell", "b": "nucleus"})
    ov = ov.sort_values("n", ascending=False).drop_duplicates("nucleus")
    ov["area"] = np.bincount(nuc.ravel())[ov["nucleus"].values]
    n_per_cell = ov.groupby("cell").size()
    multi = ov[ov["cell"].isin(n_per_cell.index[n_per_cell > 1])]
    relabel = np.arange(nuc.max() + 1)
    to_split = dict()
    for c, g in multi.groupby("cell"):
        g = g.sort_values("area", ascending=False)
        big = g[g["area"] >= min_nucleus_area]
        largest_share = g["area"].iloc[0] / g["area"].sum()
        if (len(big) >= 2) and (largest_share <= dominant_nuclei_threshold):
            to_split[c] = big["nucleus"].values
            relabel[g.loc[g["area"] < min_nucleus_area, "nucleus"].values] = 0
        else:
            relabel[g["nucleus"].values] = g["nucleus"].iloc[0]
    nuc = relabel[nuc]
    stats["cells_nuclei_merged"] = multi["cell"].nunique() - len(to_split)
    stats["cells_divided"] = len(to_split)

    if to_split:
        objs = ndi.find_objects(cell)
        next_label = cell.max() + 1
        for c, seeds in to_split.items():
            sl = objs[c - 1]
            region = cell[sl] == c
            markers = np.where(region & np.isin(nuc[sl], seeds), nuc[sl], 0)
            distance = ndi.distance_transform_edt(markers == 0)
            parts = watershed(distance, markers, mask=region)
            sub = cell[sl]
            for s in seeds:
                sub[parts == s] = next_label
                next_label += 1

    # 5. Nuclei in uncovered pixels
    stats["cells_added_from_dna"] = 0
    if fill_gaps:
        if (nuclear_image is None) or (nuclear_image.shape != cell.shape):
            print("No DNA image of matching shape given; gaps were not filled.")
        else:
            new_nuc = _detect_nuclei_in_gaps(
                nuclear_image, cell, nuc, min_nucleus_area=min_nucleus_area
            )
            if new_nuc.max() > 0:
                grown = expand_labels(new_nuc, nucleus_expansion)
                free = (grown > 0) & (cell == 0)
                cell[free] = grown[free] + cell.max()
                sel = (new_nuc > 0) & (cell > 0)
                nuc[sel] = new_nuc[sel] + nuc.max()
            stats["cells_added_from_dna"] = int(new_nuc.max())

    # Align nuclear labels to cell labels and make labels consecutive
    nuc[cell == 0] = 0
    cell = relabel_sequential(cell)[0]
    nuc = np.where(nuc > 0, cell, 0)
    stats["cells_out"] = int(cell.max())
    stats["cells_without_nucleus"] = int(cell.max() - (len(np.unique(nuc)) - 1))
    if verbose:
        print("Cell-nucleus pairing: " + ", ".join(f"{k}={v}" for k, v in stats.items()))
    return np.stack([cell, nuc], axis=-1).astype(np.uint32)


def _detect_nuclei_in_gaps(
    nuclear_image: Array,
    cell_mask: Array,
    nuclei_mask: Array,
    min_nucleus_area: int = 10,
    quantile: float = 0.1,
    sigma: float = 1.0,
    min_distance: int = 3,
) -> Array:
    """
    Detect nuclei in pixels not covered by any cell.

    Pixels are called nuclear if the smoothed DNA signal exceeds the ``quantile``
    of the mean signal of the existing nuclei; touching nuclei are separated by
    watershed on the distance transform. Objects smaller than ``min_nucleus_area``
    or larger than four times the median nuclear area are discarded.
    """
    from skimage.feature import peak_local_max
    from skimage.filters import threshold_otsu
    from skimage.segmentation import watershed, relabel_sequential

    dna = ndi.gaussian_filter(nuclear_image.astype(float), sigma)
    ids = np.unique(nuclei_mask)
    ids = ids[ids > 0]
    if len(ids) >= 10:
        threshold = np.quantile(ndi.mean(dna, labels=nuclei_mask, index=ids), quantile)
        max_area = 4 * np.median(np.bincount(nuclei_mask.ravel())[ids])
    else:
        threshold = threshold_otsu(dna)
        max_area = np.inf
    candidate = ndi.binary_opening((dna > threshold) & (cell_mask == 0))
    if not candidate.any():
        return np.zeros(cell_mask.shape, dtype=np.int64)
    dist = ndi.distance_transform_edt(candidate)
    pieces = ndi.label(candidate)[0]
    peaks = peak_local_max(
        dist, min_distance=min_distance, labels=pieces, exclude_border=False
    )
    markers = np.zeros(cell_mask.shape, dtype=np.int64)
    markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
    nuclei = watershed(-dist, markers, mask=candidate)
    area = np.bincount(nuclei.ravel())
    keep = (area >= min_nucleus_area) & (area <= max_area)
    keep[0] = False
    nuclei[~keep[nuclei]] = 0
    return relabel_sequential(nuclei)[0]


def dna_image(roi: _roi.ROI) -> Array:
    """Mean of the DNA channels of an ROI, without normalization or clipping."""
    stack, labels = roi.stack, roi.channel_labels
    if roi.channel_exclude is not None:
        keep = ~np.asarray(roi.channel_exclude, dtype=bool)
        stack, labels = stack[keep], labels[keep]
    return np.asarray(stack[nuclear_channels(labels)], dtype=float).mean(0)


def qc_cell_and_nuclear_masks(
    mask: Array,
    nuclear_image: Array,
    min_cell_area: float = 15,
    partial_nucleus_area: float = 15,
    merge_split_nuclei: bool = True,
    border_ratio: float = 0.9,
    max_union_area: float = 1.5,
    min_solidity: float = 0.9,
    min_contact: int = 3,
    verbose: bool = False,
) -> tp.Tuple[Array, pd.DataFrame]:
    """
    Quality control of paired cell and nuclear masks, as returned by
    ``align_cell_and_nuclear_masks`` (each nucleus carries the label of its cell).

    Rules, in order:
        1. A nucleus split in two is made whole again: two touching nuclei, with their
           cells, are merged when the DNA signal along their shared border (at least
           ``min_contact`` pixels) is at least ``border_ratio`` times the mean DNA
           signal of the brighter nucleus (no dip between them), and together they form
           a convex object (solidity of at least ``min_solidity``) no larger than
           ``max_union_area`` times the median nucleus.
        2. Cells smaller than ``min_cell_area`` pixels are removed.
        3. Nuclei are kept whatever their size. Nuclei smaller than
           ``partial_nucleus_area`` pixels are flagged as partial, for example a nucleus
           cut near its top or bottom by the section.

    Parameters
    ----------
    mask: Array
        Array of shape (Y, X, 2) with the cell and nuclear masks.
    nuclear_image: Array
        DNA image of shape (Y, X), without clipping (e.g. ``dna_image``).

    Returns
    -------
    The masks, shape (Y, X, 2), and a table with one row per cell: ``cell`` (label),
    ``area`` and ``nucleus_area`` (pixels), ``has_nucleus``, ``partial_nucleus`` and
    ``merged_pieces`` (number of segmented cells merged into this one).
    """
    from skimage.morphology import convex_hull_image
    from skimage.segmentation import relabel_sequential

    if (mask.ndim != 3) or (mask.shape[-1] != 2):
        raise ValueError("Mask must have shape (Y, X, 2) with cell and nuclear masks.")
    cell = mask[..., 0].astype(np.int64)
    nuc = np.where(mask[..., 1] > 0, cell, 0)
    target = np.arange(cell.max() + 1)
    n_merged = 0

    # 1. Nuclei split in two
    if merge_split_nuclei and nuc.max() > 0:
        dna = ndi.gaussian_filter(np.asarray(nuclear_image, dtype=float), 1.0)
        ids = np.unique(nuc)
        ids = ids[ids > 0]
        mean_dna = np.zeros(nuc.max() + 1)
        mean_dna[ids] = ndi.mean(dna, labels=nuc, index=ids)
        area = np.bincount(nuc.ravel(), minlength=nuc.max() + 1)
        median_area = np.median(area[ids])
        border = list()
        for a, b, dy, dx in [(nuc[:, :-1], nuc[:, 1:], 0, 1), (nuc[:-1, :], nuc[1:, :], 1, 0)]:
            sel = (a > 0) & (b > 0) & (a != b)
            y, x = np.nonzero(sel)
            border.append(
                pd.DataFrame(
                    {
                        "a": np.minimum(a[sel], b[sel]),
                        "b": np.maximum(a[sel], b[sel]),
                        "dna": (dna[y, x] + dna[y + dy, x + dx]) / 2,
                    }
                )
            )
        pairs = pd.concat(border).groupby(["a", "b"])["dna"].agg(["mean", "size"])
        pairs = pairs.reset_index()
        a, b = pairs["a"].values, pairs["b"].values
        pairs["ratio"] = pairs["mean"].values / np.maximum(mean_dna[a], mean_dna[b])
        pairs = pairs[
            (pairs["size"] >= min_contact)
            & (pairs["ratio"] >= border_ratio)
            & (area[a] + area[b] <= max_union_area * median_area)
        ].sort_values("ratio", ascending=False)
        objs = ndi.find_objects(nuc)
        used = set()
        for a, b in zip(pairs["a"].values, pairs["b"].values):
            if (a in used) or (b in used):
                continue
            sl = tuple(
                slice(min(p.start, q.start), max(p.stop, q.stop))
                for p, q in zip(objs[a - 1], objs[b - 1])
            )
            union = np.isin(nuc[sl], [a, b])
            if union.sum() / convex_hull_image(union).sum() < min_solidity:
                continue
            keep, drop = (a, b) if area[a] >= area[b] else (b, a)
            target[drop] = keep
            used.update([a, b])
            n_merged += 1
        cell = target[cell]
        nuc = target[nuc]
    pieces = np.bincount(target, minlength=len(target))

    # 2. Small cells
    cell_area = np.bincount(cell.ravel(), minlength=cell.max() + 1)
    small = (cell_area > 0) & (cell_area < min_cell_area)
    small[0] = False
    cell[small[cell]] = 0
    nuc[cell == 0] = 0

    # Labels made consecutive; nuclei take the label of their cell
    new = relabel_sequential(cell)[0]
    nuc = np.where(nuc > 0, new, 0)
    old_of_new = np.zeros(new.max() + 1, dtype=np.int64)
    old_of_new[new[new > 0]] = cell[new > 0]
    ids = np.arange(1, new.max() + 1)
    area = np.bincount(new.ravel(), minlength=new.max() + 1)[ids]
    nuc_area = np.bincount(nuc.ravel(), minlength=new.max() + 1)[ids]
    table = pd.DataFrame(
        {
            "cell": ids,
            "area": area,
            "nucleus_area": nuc_area,
            "has_nucleus": nuc_area > 0,
            "partial_nucleus": (nuc_area > 0) & (nuc_area < partial_nucleus_area),
            "merged_pieces": pieces[old_of_new[ids]],
        }
    )
    if verbose:
        print(
            f"Cell QC: {n_merged} nuclei split in two were merged, {int(small.sum())} cells "
            f"under {min_cell_area:g} px removed; {int((~table['has_nucleus']).sum())} cells "
            f"without nucleus, {int(table['partial_nucleus'].sum())} with a partial nucleus."
        )
    return np.stack([new, nuc], axis=-1).astype(np.uint32), table


def segment_roi(
    roi: _roi.ROI,
    from_probabilities: bool = False,
    model: tp.Union[tp.Literal["deepcell"], tp.Literal["stardist"]] = "deepcell",
    compartment: str = "nuclear",
    postprocessing: bool = True,
    save: bool = True,
    overwrite: bool = True,
    plot_segmentation: bool = True,
    verbose: bool = False,
    image_mpp: float = 1.0,
    membrane_channels: tp.Sequence[str] = None,
    postprocessing_mode: str = "align",
    fill_gaps: bool = False,
    output_mask_suffix: str = "",
    membrane_mode: str = "max",
    whole_cell_threshold: float = 0.05,
    min_cell_area: float = 15.0,
    min_nucleus_area: float = None,
    qc: bool = True,
    partial_nucleus_area: float = 15.0,
    save_maps: bool = False,
    output_dir: Path = None,
) -> tp.Dict[str, tp.Any]:
    """
    Segment the area of an ROI.

    Parameters
    ----------
    roi: ROI
        ROI object to process.
    from_probabilities: bool
        Whether to use the probability array instead of the channel stack.
        If true, only the "deepcell" ``model`` can be used.
    model: str
        One of ``deepcell``, ``stardist`` or ``cellpose``.
    compartment: str
        One of ``nuclear``, ``cytoplasm``, or ``both``.
    postprocessing: bool
        Whether to pair cell and nuclear masks when ``compartment`` is ``both``.
    save: bool
        Whether to save the outputs to disk.
    overwrite: bool
        Whether to overwrite files when ``save`` is True.
    plot_segmentation: bool
        Whether to make a figure illustrating the segmentation.
    image_mpp: float
        Pixel size of the stack in micrometres (1.0 for IMC), passed to DeepCell.
    membrane_channels: list[str]
        Markers combined into the membrane image when segmenting from the stack.
        See ``prepare_stack``.
    postprocessing_mode: str
        ``align`` (``align_cell_and_nuclear_masks``) or ``legacy``
        (``deepcell_postprocess_both_compartments``).
    fill_gaps: bool
        Whether to add cells for nuclei found in the DNA channels outside all cells.
        Only used with ``postprocessing_mode`` ``align``.
    output_mask_suffix: str
        Suffix added to the mask file names, e.g. '_v2' writes '<roi>_full_mask_v2.tiff'.
    membrane_mode: str
        How membrane channels are combined into the membrane image (``max`` or
        ``mean``), see ``prepare_stack``.
    whole_cell_threshold: float
        Threshold on Mesmer's predicted cell interior probability. Lower values give
        larger cells; Mesmer's default (0.2) draws outlines close to the nucleus on IMC
        images. ``None`` keeps Mesmer's default.
    min_cell_area: float
        Smallest cell kept, in um2, applied in Mesmer and after pairing. ``None`` or 0
        keeps Mesmer's default (15 pixels at 0.5 um/px, about 4 um2).
    min_nucleus_area: float
        Smallest nucleus kept by Mesmer, in um2. ``None`` keeps Mesmer's default.
    qc: bool
        Whether to apply ``qc_cell_and_nuclear_masks`` after pairing (``align`` mode):
        nuclei split in two are merged, cells below ``min_cell_area`` are removed and a
        per-cell QC table is made.
    partial_nucleus_area: float
        Nuclei smaller than this (um2) are flagged as partial in the QC table.
    save_maps: bool
        Whether to save Mesmer's per-pixel outputs (``MESMER_MAPS``) and its input
        images (``model`` deepcell, ``compartment`` both, from the stack).
    output_dir: Path
        Directory for the outputs. By default they are written next to the stack.

    Returns
    -------
    Dictionary with the masks (``cell`` and/or ``nuclei``) and, when made, the QC table
    (``cell_qc``) and Mesmer's per-pixel outputs (``mesmer_maps``).
    """
    assert postprocessing_mode in ["align", "legacy"]
    kw_cell, kw_nuc = mesmer_postprocess_kwargs(
        whole_cell_threshold, min_cell_area, min_nucleus_area
    )
    maps = None
    if from_probabilities:
        assert model in ["deepcell"]
        image = roi.probabilities
        # probabilities are predicted on a 2x upscaled image: return to the stack grid
        shape = tuple([int(x / 2) for x in image.shape[1:]])
        image = resize(image / image.max(), (3,) + shape)
        mask = deepcell_segment_probabilities(
            np.moveaxis(image, 0, -1),
            compartment,
            image_mpp=image_mpp,
            postprocess_kwargs_whole_cell=kw_cell,
            postprocess_kwargs_nuclear=kw_nuc,
        )
    else:
        assert model in ["deepcell", "stardist", "cellpose"]
        image = prepare_stack(
            roi.stack,
            roi.channel_labels,
            compartment,
            roi.channel_exclude,
            membrane_channels=membrane_channels,
            normalization="percentile" if model == "deepcell" else "equalize",
            membrane_mode=membrane_mode,
        )
        if model == "stardist":
            assert compartment == "nuclear"
            mask = stardist_segment_nuclei(image)

        elif model == "deepcell":
            mask = deepcell_segment(
                to_deepcell_input(image),
                compartment=compartment,
                image_mpp=image_mpp,
                postprocess_kwargs_whole_cell=kw_cell,
                postprocess_kwargs_nuclear=kw_nuc,
                return_maps=save_maps and compartment == "both",
            )
            if isinstance(mask, tuple):
                mask, maps = mask
        elif model == "cellpose":
            mask = cellpose_segment(image, compartment)

    cell_qc = None
    if postprocessing and compartment in ["both"]:
        if postprocessing_mode == "legacy":
            mask = deepcell_postprocess_both_compartments(
                mask,
                plot=False,
                roi=roi,
                fig_output_prefix=roi.get_input_filename("stack").replace_(
                    ".tiff", "_segmentation."
                ),
                verbose=verbose,
            )
        else:
            dna = None
            if fill_gaps:
                dna = prepare_stack(
                    roi.stack,
                    roi.channel_labels,
                    "nuclear",
                    roi.channel_exclude,
                    normalization="percentile",
                )
            mask = align_cell_and_nuclear_masks(
                mask, nuclear_image=dna, fill_gaps=fill_gaps, verbose=verbose
            )
            if qc:
                px = image_mpp**2
                mask, cell_qc = qc_cell_and_nuclear_masks(
                    mask,
                    dna_image(roi),
                    min_cell_area=(min_cell_area or 0) / px,
                    partial_nucleus_area=(partial_nucleus_area or 0) / px,
                    verbose=verbose,
                )
                cell_qc.insert(2, "area_um2", cell_qc["area"] * px)
                cell_qc.insert(4, "nucleus_area_um2", cell_qc["nucleus_area"] * px)

    mask_dict = dict()
    if compartment == "both":
        for i, comp in enumerate(["cell", "nuclei"]):
            mask_dict[comp] = mask[..., i]
    elif compartment == "nuclear":
        mask_dict["nuclei"] = mask
    elif compartment == "cytoplasm":
        mask_dict["cell"] = mask

    def output_file(file: Path, tag: str = "", ext: str = ".tiff") -> Path:
        file = file.replace_(".tiff", tag + output_mask_suffix + ext)
        return Path(output_dir) / file.name if output_dir is not None else file

    if save:
        if output_dir is not None:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        for comp, mask in mask_dict.items():
            mask_file = output_file(roi.get_input_filename(comp + "_mask"))
            if overwrite or (not overwrite and not mask_file.exists()):
                tifffile.imwrite(mask_file, mask)
        stack_file = roi.get_input_filename("stack")
        if cell_qc is not None:
            cell_qc.to_csv(output_file(stack_file, "_cell_qc", ".csv"), index=False)
        if maps is not None:
            tifffile.imwrite(
                output_file(stack_file, "_mesmer_maps"),
                maps,
                photometric="minisblack",
                compression="zlib",
                metadata={"axes": "CYX", "Channel": MESMER_MAPS},
            )
            tifffile.imwrite(
                output_file(stack_file, "_mesmer_input"),
                np.asarray(image, dtype=np.float32),
                photometric="minisblack",
                compression="zlib",
                metadata={"axes": "CYX", "Channel": ["nuclear", "membrane"]},
            )

    if plot_segmentation:
        fig = plot_image_and_mask(image, mask_dict)
        if save:
            fig_file = output_file(
                roi.get_input_filename("stack"),
                f"_segmentation_{model}_{compartment}",
                ".svg",
            )
            if overwrite or (not overwrite and not fig_file.exists()):
                fig.savefig(fig_file, dpi=300, bbox_inches="tight")

    out = dict(mask_dict)
    if cell_qc is not None:
        out["cell_qc"] = cell_qc
    if maps is not None:
        out["mesmer_maps"] = maps
    return out


# def segment_decomposition():
#     """Some experiments with tensor decomposition"""
#     import tensorly as tl
#     from tensorly.decomposition import tucker, non_negative_tucker, parafac

#     from imc.demo import generate_disk_masks, generate_stack

#     # Synthetic data
#     Y = np.asarray([generate_disk_masks()[np.newaxis, ...] for _ in range(100)])
#     X = np.asarray([generate_stack(m.squeeze()) for m in Y])

#     fig, axes = plt.subplots(1, 3)
#     for ax, a in zip(axes, X[0]):
#         ax.imshow(a)

#     core, factors = tucker(X[0], [1, 128, 128])
#     core, factors = non_negative_tucker(X[0], [1, 128, 128])
#     reconstruction = tl.tucker_to_tensor((core, factors))

#     fig, axes = plt.subplots(1, 5)
#     for ax, a in zip(axes, X[0]):
#         ax.imshow(a)
#     q = factors[1] @ factors[2].T
#     axes[-2].imshow(q)
#     axes[-1].imshow(q > 1)

#     factors[0]  # samples
#     factors[1]  # channels

#     # if decomposing in more than one factor:
#     fig, axes = plt.subplots(1, 4)
#     for ax, dim in zip(axes, factors):
#         ax.scatter(*dim.T, s=2, alpha=0.5)

#     fig, axes = plt.subplots(1, 4)
#     for ax, dim in zip(axes, factors):
#         ax.plot(dim)

#     # Real data
#     r = prj.rois[50]

#     exc = r.channel_exclude.index.str.contains("EMPTY|80|129")
#     stack = normalize(r.stack[~exc])

#     core, factors = tucker(stack, [3] + list(r.shape[1:]))
#     reconstruction = tl.tucker_to_tensor((core, factors))

#     fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
#     axes[0][0].imshow(stack.mean(0))
#     axes[0][1].imshow(reconstruction.mean(0))
#     axes[1][0].imshow(stack[-1])
#     v = np.absolute(reconstruction[-1]).max()
#     axes[1][1].imshow(reconstruction[-1], cmap="RdBu_r", vmin=-v, vmax=v)  # .clip(min=0))

#     channel_imp = pd.Series(factors[0].squeeze(), index=r.channel_labels[~exc])
#     channel_mean = pd.Series(stack.mean((1, 2)), index=r.channel_labels[~exc])

#     plt.scatter(channel_mean, channel_imp)

#     fig, axes = plt.subplots(1, 8)
#     for ax, a in zip(axes, X[0]):
#         ax.imshow(a)
#     q = factors[1] @ factors[2].T
#     axes[-2].imshow(q)
#     axes[-1].imshow(q > 1)

#     probs = zoom(r.probabilities, (1, 0.5, 0.5))
#     probs = probs / probs.max()
#     core, factors = tucker(probs, [3] + list(r.shape[1:]))
#     reconstruction = tl.tucker_to_tensor((core, factors))

#     fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)
#     axes[0].imshow(np.moveaxis(probs, 0, -1))
#     axes[1].imshow(np.moveaxis(reconstruction, 0, -1))

#     weights, factors = parafac(probs, rank=50, init="random", tol=10e-6)
#     cp_reconstruction = tl.cp_to_tensor((weights, factors))

#     plt.imshow(np.moveaxis(minmax_scale(cp_reconstruction), 0, -1))

#     def PCA2D_2D(samples, row_top, col_top):
#         """samples are 2d matrices"""
#         size = samples[0].shape
#         # m*n matrix
#         mean = np.zeros(size)

#         for s in samples:
#             mean = mean + s

#         # get the mean of all samples
#         mean /= float(len(samples))

#         # n*n matrix
#         cov_row = np.zeros((size[1], size[1]))
#         for s in samples:
#             diff = s - mean
#             cov_row = cov_row + np.dot(diff.T, diff)
#         cov_row /= float(len(samples))
#         row_eval, row_evec = np.linalg.eig(cov_row)
#         # select the top t evals
#         sorted_index = np.argsort(row_eval)
#         # using slice operation to reverse
#         X = row_evec[:, sorted_index[: -row_top - 1 : -1]]

#         # m*m matrix
#         cov_col = np.zeros((size[0], size[0]))
#         for s in samples:
#             diff = s - mean
#             cov_col += np.dot(diff, diff.T)
#         cov_col /= float(len(samples))
#         col_eval, col_evec = np.linalg.eig(cov_col)
#         sorted_index = np.argsort(col_eval)
#         Z = col_evec[:, sorted_index[: -col_top - 1 : -1]]

#         return X, Z


# def _segment_probabilities(
#     min_distance=10,
#     detection_threshold=0.1,
#     distance_threshold=0.01,
#     exclude_border=False,
#     small_objects_threshold=0
# ):
#     from skimage.feature import peak_local_max
#     from skimage.measure import label
#     from skimage.morphology import (
#         watershed,
#         remove_small_objects,
#         h_maxima,
#         disk,
#         square,
#         dilation,
#     )
#     from skimage.segmentation import relabel_sequential
#     from deepcell_toolbox.utils import erode_edges, fill_holes

#     probs = normalize(probs, "yxc")

#     inner_distance = probs[..., 0]
#     outer_distance = probs[..., 1]

#     coords = peak_local_max(
#         inner_distance,
#         min_distance=min_distance,
#         threshold_abs=detection_threshold,
#         exclude_border=exclude_border,
#     )

#     markers = np.zeros(inner_distance.shape)
#     markers[coords[:, 0], coords[:, 1]] = 1
#     markers = label(markers)
#     label_image = watershed(
#         -outer_distance, markers, mask=outer_distance > distance_threshold
#     )
#     label_image = erode_edges(label_image, 1)

#     # Remove small objects
#     label_image = remove_small_objects(
#         label_image, min_size=small_objects_threshold
#     )

#     # Relabel the label image
#     label_image, _, _ = relabel_sequential(label_image)

#     label_images.append(label_image)


#     interior_model_smooth = 1

#     interior_batch = probs[..., 0]
#     interior_batch = nd.gaussian_filter(interior_batch, interior_model_smooth)

#     if pixel_expansion is not None:
#         interior_batch = dilation(interior_batch, selem=square(pixel_expansion * 2 + 1))

#     maxima_batch = maxima_predictions[batch, ..., 0]
#     maxima_batch = nd.gaussian_filter(maxima_batch, maxima_model_smooth)

#     markers = h_maxima(image=maxima_batch,
#                        h=maxima_threshold,
#                        selem=disk(radius))

#     markers = label(markers)

#     label_image = watershed(-interior_batch,
#                             markers,
#                             mask=interior_batch > interior_threshold,
#                             watershed_line=0)

#     # Remove small objects
#     label_image = remove_small_objects(label_image, min_size=small_objects_threshold)

#     # fill in holes that lie completely within a segmentation label
#     if fill_holes_threshold > 0:
#         label_image = fill_holes(label_image, size=fill_holes_threshold)

#     # Relabel the label image
#     label_image, _, _ = relabel_sequential(label_image)
