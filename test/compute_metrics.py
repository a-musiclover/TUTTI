import argparse
import json
import logging
import os
from pathlib import Path

from pyMV2H.converter.midi_converter import MidiConverter as Converter
from pyMV2H.metrics.mv2h import mv2h
from pyMV2H.utils.music import Music
from pyMV2H.utils.mv2h import MV2H
from tqdm import tqdm

TEST_DIR = Path(__file__).resolve().parent
FINETUNE_DIR = TEST_DIR.parent / "finetune"
TEST_OUTPUT_ROOT = TEST_DIR / "outputs"
DEFAULT_MIDI_DIR = TEST_OUTPUT_ROOT / "midi_outputs"
DEFAULT_TXT_DIR = TEST_OUTPUT_ROOT / "txt_outputs"
DEFAULT_RESULTS_DIR = TEST_OUTPUT_ROOT / "results"

import sys
if str(FINETUNE_DIR) not in sys.path:
    sys.path.insert(0, str(FINETUNE_DIR))

from config import *


def compute_metrics_from_midi():
    """Compute MV2H metrics from generated MIDI files."""
    return compute_metrics_from_midi_dirs()


def compute_metrics_from_midi_dirs(midi_dir=None, txt_dir=None, results_dir=None):
    """Compute MV2H metrics from generated MIDI files."""
    midi_dir = Path(midi_dir or DEFAULT_MIDI_DIR)
    txt_dir = Path(txt_dir or DEFAULT_TXT_DIR)
    results_dir = Path(results_dir or DEFAULT_RESULTS_DIR)

    logging.info("=" * 60)
    logging.info("Computing MV2H metrics from MIDI files...")
    logging.info("=" * 60)

    os.makedirs(txt_dir, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    global_res_dict = MV2H(multi_pitch=0, voice=0, meter=0, harmony=0, note_value=0)
    individual_results = []
    error_list = []
    counter = 0

    predicted_midi_files = list(midi_dir.glob("*_predicted.mid"))
    logging.info(f"Found {len(predicted_midi_files)} predicted MIDI files")

    for predicted_midi_path in tqdm(predicted_midi_files, desc="Computing MV2H"):
        stem = predicted_midi_path.stem.replace("_predicted", "")

        true_midi_path = midi_dir / f"{stem}_true.mid"
        true_txt_path = txt_dir / f"{stem}_true.txt"
        predicted_txt_path = txt_dir / f"{stem}_predicted.txt"

        if not true_midi_path.exists():
            logging.warning(f"Missing true MIDI: {true_midi_path}")
            continue

        try:
            if not true_txt_path.exists():
                Converter(file=str(true_midi_path), output=str(true_txt_path)).convert_file()

            if not predicted_txt_path.exists():
                Converter(file=str(predicted_midi_path), output=str(predicted_txt_path)).convert_file()

            reference = Music.from_file(str(true_txt_path))
            transcription = Music.from_file(str(predicted_txt_path))
            result = mv2h(reference, transcription)

            individual_results.append({
                "file_name": stem,
                "multi-pitch": result.__multi_pitch__,
                "voice": result.voice,
                "meter": result.meter,
                "harmony": result.harmony,
                "note_value": result.note_value,
                "mv2h": result.mv2h,
            })

            global_res_dict.__multi_pitch__ += result.__multi_pitch__
            global_res_dict.__voice__ += result.voice
            global_res_dict.__meter__ += result.meter
            global_res_dict.__harmony__ += result.harmony
            global_res_dict.__note_value__ += result.note_value
            counter += 1

        except Exception as error:
            error_list.append(stem)
            logging.error(f"Failed to compute MV2H for {stem}: {error}")

    if counter > 0:
        mv2h_dict = {
            "multi-pitch": global_res_dict.__multi_pitch__ / counter,
            "voice": global_res_dict.__voice__ / counter,
            "meter": global_res_dict.__meter__ / counter,
            "harmony": global_res_dict.__harmony__ / counter,
            "note_value": global_res_dict.__note_value__ / counter,
            "mv2h": global_res_dict.mv2h / counter,
        }

        logging.info("=" * 60)
        logging.info("Final MV2H results:")
        for metric, value in mv2h_dict.items():
            logging.info(f"  {metric}: {value:.4f}")
        logging.info(f"Successfully processed: {counter}/{len(predicted_midi_files)} files")
        logging.info("=" * 60)

        with open(results_dir / "mv2h_results.json", "w", encoding="utf-8") as file:
            json.dump(mv2h_dict, file, indent=2)

        with open(results_dir / "mv2h_results_individual.json", "w", encoding="utf-8") as file:
            json.dump(individual_results, file, indent=4, ensure_ascii=False)

        logging.info(f"Results saved to: {results_dir / 'mv2h_results.json'}")
        logging.info(f"Per-file detailed results saved to: {results_dir / 'mv2h_results_individual.json'}")
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
            json.dump(mv2h_dict, file, indent=2)
        with open(results_dir / "mv2h_results_individual.json", "w", encoding="utf-8") as file:
            json.dump(individual_results, file, indent=4, ensure_ascii=False)
        logging.warning("No files were processed successfully; wrote zero-valued results.")

    if error_list:
        with open(results_dir / "error_files.txt", "w", encoding="utf-8") as file:
            file.write("\n".join(error_list))
        logging.warning(f"{len(error_list)} errors recorded")


def main():
    parser = argparse.ArgumentParser(description="Compute MV2H metrics from predicted MIDI files")
    parser.parse_args()
    compute_metrics_from_midi()


if __name__ == "__main__":
    main()