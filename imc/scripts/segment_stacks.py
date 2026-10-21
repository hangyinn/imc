#!/usr/bin/env python

"""
Segment image stacks.
"""

import sys
import argparse
import typing as tp
from dataclasses import dataclass

import numpy as np
import pandas as pd
import tifffile
import matplotlib.pyplot as plt

from imc import ROI
from imc.types import Path, Series, Array
from imc.segmentation import segment_roi, plot_image_and_mask
from imc.scripts import build_cli, find_tiffs


def main(cli: tp.Sequence[str] = None) -> int:
    parser = build_cli("segment")
    args = parser.parse_args(cli)
    if len(args.tiffs) == 0:
        args.tiffs = find_tiffs()
        if len(args.tiffs) == 0:
            print("TIFF files were not provided and could not be found!")
            return 1

    fs = "\n\t- " + "\n\t- ".join([f.as_posix() for f in args.tiffs])
    print(f"Starting segmentation step for {len(args.tiffs)} TIFF files:{fs}!")

    # Prepare ROI objects
    rois = list()
    for tiff in args.tiffs:
        roi = ROI.from_stack(tiff)
        roi.set_channel_exclude(args.channel_exclude.split(","))
        rois.append(roi)

    membrane_channels = [c for c in args.membrane_channels.split(",") if c]
    layers = {"both": ["cell", "nuclei"], "nuclear": ["nuclei"], "cytoplasm": ["cell"]}

    # Run segmentation
    for roi in rois:
        mask_files = {
            layer: roi.get_input_filename(layer + "_mask").replace_(
                ".tiff", args.output_mask_suffix + ".tiff"
            )
            for layer in layers[args.compartment]
        }
        if args.output_dir is not None:
            mask_files = {k: args.output_dir / v.name for k, v in mask_files.items()}
        exists = all(f.exists() for f in mask_files.values())
        if exists and not args.overwrite:
            print(f"Mask for '{roi}' already exists, skipping...")
            continue

        print(f"Started segmentation of '{roi} with shape: '{roi.stack.shape}'")
        try:
            _ = segment_roi(
                roi,
                from_probabilities=args.from_probabilities,
                model=args.model,
                compartment=args.compartment,
                postprocessing=args.postprocessing,
                save=args.save,
                overwrite=args.overwrite,
                plot_segmentation=args.plot,
                verbose=not args.quiet,
                image_mpp=args.image_mpp,
                membrane_channels=membrane_channels,
                postprocessing_mode=args.postprocessing_mode,
                fill_gaps=args.fill_gaps,
                output_mask_suffix=args.output_mask_suffix,
                membrane_mode=args.membrane_mode,
                whole_cell_threshold=args.whole_cell_threshold,
                cell_expansion=args.cell_expansion,
                min_cell_area=args.min_cell_area,
                min_nucleus_area=args.min_nucleus_area,
                qc=args.qc,
                partial_nucleus_area=args.partial_nucleus_area,
                save_maps=args.save_maps,
                output_dir=args.output_dir,
                membrane_source=args.membrane_source,
                ilastik_dir=args.ilastik_dir,
                keep_nuclei_whole=args.keep_nuclei_whole,
                merge_marker=args.merge_marker,
                max_marker_cell_area=args.max_marker_cell_area,
            )
        except ValueError as e:
            print("Error segmenting stack. Perhaps XY shape is not compatible?")
            print(e)
            continue
        print(f"Finished segmentation of '{roi}'.")

    print("Finished segmentation step!")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(1)
