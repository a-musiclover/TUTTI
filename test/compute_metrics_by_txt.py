import argparse
import json
import logging
import os
from pathlib import Path

from pyMV2H.metrics.mv2h import mv2h
from pyMV2H.utils.mv2h import MV2H
from pyMV2H.utils.music import Music
from tqdm import tqdm

TEST_DIR = Path(__file__).resolve().parent
FINETUNE_DIR = TEST_DIR.parent / "finetune"
TEST_OUTPUT_ROOT = TEST_DIR / "outputs"
DEFAULT_TXT_DIR = TEST_OUTPUT_ROOT / "txt_outputs"
DEFAULT_RESULTS_DIR = TEST_OUTPUT_ROOT / "results"

import sys
if str(FINETUNE_DIR) not in sys.path:
    sys.path.insert(0, str(FINETUNE_DIR))

from config import *


def compute_metrics_from_txt():
    """Compute MV2H metrics directly from paired TXT files."""
    compute_metrics_from_txt_dirs()


def compute_metrics_from_txt_dirs(txt_dir=None, results_dir=None):
    """Compute MV2H metrics directly from paired TXT files."""
    results_dir = Path(results_dir or DEFAULT_RESULTS_DIR)
    txt_dir = Path(txt_dir or DEFAULT_TXT_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"Scanning directory: {txt_dir} ...")
    true_files = list(txt_dir.glob("*_true.txt"))

    global_res_dict = MV2H(multi_pitch=0, voice=0, meter=0, harmony=0, note_value=0)
    counter = 0
    error_list = []
    individual_results = []

    if not true_files:
        logging.error(f"No *_true.txt files found in {txt_dir}")
        mv2h_dict = {
            "multi-pitch": 0,
            "voice": 0,
            "meter": 0,
            "harmony": 0,
            "note_value": 0,
            "mv2h": 0,
        }
        with open(results_dir / "mv2h_results.json", "w", encoding="utf-8") as file:
            json.dump(mv2h_dict, file, indent=4)
        with open(results_dir / "mv2h_results_individual.json", "w", encoding="utf-8") as file:
            json.dump([], file, indent=4, ensure_ascii=False)
        return

    logging.info(f"Found {len(true_files)} reference files, starting computation...")

    for true_txt_path in tqdm(true_files, desc="Computing MV2H"):
        base_name = ""
        try:
            base_name = true_txt_path.name.replace("_true.txt", "")
            predicted_txt_path = txt_dir / f"{base_name}_predicted.txt"

            if not predicted_txt_path.exists():
                continue

            reference_file = Music.from_file(str(true_txt_path))
            transcription_file = Music.from_file(str(predicted_txt_path))
            result_dict = mv2h(reference_file, transcription_file)

            global_res_dict.__multi_pitch__ += result_dict.__multi_pitch__
            global_res_dict.__voice__ += result_dict.voice
            global_res_dict.__meter__ += result_dict.meter
            global_res_dict.__harmony__ += result_dict.harmony
            global_res_dict.__note_value__ += result_dict.note_value
            counter += 1

            individual_results.append({
                "file_name": base_name,
                "multi-pitch": result_dict.__multi_pitch__,
                "voice": result_dict.voice,
                "meter": result_dict.meter,
                "harmony": result_dict.harmony,
                "note_value": result_dict.note_value,
                "mv2h": result_dict.mv2h,
            })

        except Exception as error:
            logging.error(f"Failed to compute MV2H for {base_name}: {error}")
            error_list.append(base_name)

    if error_list:
        with open(results_dir / "error_files.txt", "w", encoding="utf-8") as error_file:
            error_file.write("\n".join(error_list))
        logging.warning(f"{len(error_list)} errors recorded in {results_dir / 'error_files.txt'}")

    if counter > 0:
        mv2h_dict = {
            "multi-pitch": global_res_dict.__multi_pitch__ / counter,
            "voice": global_res_dict.__voice__ / counter,
            "meter": global_res_dict.__meter__ / counter,
            "harmony": global_res_dict.__harmony__ / counter,
            "note_value": global_res_dict.__note_value__ / counter,
            "mv2h": global_res_dict.mv2h / counter,
        }

        with open(results_dir / "mv2h_results.json", "w", encoding="utf-8") as file:
            json.dump(mv2h_dict, file, indent=4)

        with open(results_dir / "mv2h_results_individual.json", "w", encoding="utf-8") as file:
            json.dump(individual_results, file, indent=4, ensure_ascii=False)

        logging.info(f"Processed {counter} file pairs")
        logging.info(f"Final MV2H results: {mv2h_dict}")
    else:
        mv2h_dict = {
            "multi-pitch": 0,
            "voice": 0,
            "meter": 0,
            "harmony": 0,
            "note_value": 0,
            "mv2h": 0,
        }
        with open(results_dir / "mv2h_results.json", "w", encoding="utf-8") as file:
            json.dump(mv2h_dict, file, indent=4)
        with open(results_dir / "mv2h_results_individual.json", "w", encoding="utf-8") as file:
            json.dump(individual_results, file, indent=4, ensure_ascii=False)
        logging.warning("No valid file pairs were processed; wrote zero-valued results!")


def main():
    parser = argparse.ArgumentParser(description="Compute MV2H metrics directly from TXT files")
    parser.parse_args()
    compute_metrics_from_txt()


if __name__ == "__main__":
    main()