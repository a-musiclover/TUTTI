import os
import sys
import logging
import subprocess
import json
from pathlib import Path
from tqdm import tqdm
import torch.multiprocessing as mp

TEST_DIR = Path(__file__).resolve().parent
FINETUNE_DIR = TEST_DIR.parent / "finetune"
ABCMIDI_BIN = TEST_DIR.parent / "abcmidi" / "abc2midi"
TEST_OUTPUT_ROOT = TEST_DIR / "outputs"
ABC_OUTPUT_DIR = TEST_OUTPUT_ROOT / "abc_outputs"
MIDI_OUTPUT_DIR = TEST_OUTPUT_ROOT / "midi_outputs"
TXT_OUTPUT_DIR = TEST_OUTPUT_ROOT / "txt_outputs"
TEST_RESULTS_DIR = TEST_OUTPUT_ROOT / "results"
if str(TEST_DIR) in sys.path:
    sys.path.remove(str(TEST_DIR))
sys.path.insert(0, str(TEST_DIR))

if str(FINETUNE_DIR) in sys.path:
    sys.path.remove(str(FINETUNE_DIR))
sys.path.insert(1, str(FINETUNE_DIR))

from batch_test import parallel_inference
from config import *
from compute_metrics import compute_metrics_from_midi_dirs

# Keep test outputs isolated from finetune/pretrain outputs.
TEST_OUTPUT_ROOT = TEST_DIR / "outputs"
ABC_OUTPUT_DIR = TEST_OUTPUT_ROOT / "abc_outputs"
MIDI_OUTPUT_DIR = TEST_OUTPUT_ROOT / "midi_outputs"
TXT_OUTPUT_DIR = TEST_OUTPUT_ROOT / "txt_outputs"
TEST_RESULTS_DIR = TEST_OUTPUT_ROOT / "results"

"""
# 1. Run the full pipeline
python test_pipeline_new_asap.py --gpu 0 --workers 4

# 2. Skip ABC generation and compute metrics directly if ABC files already exist
python test_pipeline_new_asap.py --skip-abc

# 3. Recompute only MV2H metrics when MIDI files already exist
python test_pipeline_new_asap.py --only-metrics

# 4. Skip MIDI generation
python test_pipeline_new_asap.py --skip-midi
"""


NUM_WORKERS = 12

# Logging configuration
os.makedirs(TEST_RESULTS_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(str(Path(TEST_RESULTS_DIR) / 'pretrained_test_pipeline.log'), mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def step1_generate_abc(gpu_id=0, num_workers=7):
    """Step 1: generate ABC files."""
    logger.info("=" * 60)
    logger.info("Step 1: starting ABC generation...")
    logger.info("=" * 60)

    results = parallel_inference(
        TEST_DATA_PATH,
        ABC_OUTPUT_DIR,
        gpu_id=gpu_id,
        num_workers=num_workers
    )

    success_count = sum(1 for r in results if r[2] == 'success')
    logger.info(f"✅ ABC generation complete: {success_count}/{len(results)} succeeded")
    return success_count


def step2_add_x1_header():
    """Step 2: add an X:1 header to every ABC file."""
    logger.info("=" * 60)
    logger.info("Step 2: adding X:1 headers...")
    logger.info("=" * 60)

    abc_files = list(Path(ABC_OUTPUT_DIR).glob("*.abc"))

    for abc_file in tqdm(abc_files, desc="Adding X:1"):
        try:
            with open(abc_file, 'r', encoding='utf-8') as f:
                content = f.read()

            if not content.startswith('X:1'):
                with open(abc_file, 'w', encoding='utf-8') as f:
                    f.write('X:1\n' + content)
        except Exception as e:
            logger.error(f"Failed to process {abc_file.name}: {e}")

    logger.info(f"✅ X:1 header added: processed {len(abc_files)} files")


def step3_build_midis():
    """Step 3: build MIDI files for both reference and prediction."""
    logger.info("=" * 60)
    logger.info("Step 3: generating MIDI files...")
    logger.info("=" * 60)

    os.makedirs(MIDI_OUTPUT_DIR, exist_ok=True)

    success_count = 0
    error_count = 0

    with open(TEST_DATA_PATH, 'r', encoding='utf-8') as file:
        tasks = [json.loads(line.strip()) for line in file]

    for data in tqdm(tasks, desc="Generating MIDI"):
        try:
            file_name = os.path.splitext(os.path.basename(data['spectrogram']))[0].replace('_spectrogram', '') + '.abc'
            predicted_file_name = os.path.splitext(os.path.basename(data['spectrogram']))[0].replace('_spectrogram', '_reduced') + '.abc'

            true_file_path = data.get('output')
            if true_file_path is None:
                true_path = os.path.join(Path(data['spectrogram']).parent.parent.parent, 'abc_07_voice_emptied_addedX1/asap')
                file_name_no_performer = file_name.rsplit('#', 1)[0] + '.abc'
                true_file_path = os.path.join(true_path, file_name_no_performer)
            else:
                true_file_path = resolve_path(true_file_path)

            predicted_path = os.path.join(ABC_OUTPUT_DIR, predicted_file_name)

            output_name = os.path.splitext(os.path.basename(data['spectrogram']))[0].replace('_spectrogram', '')

            if os.path.exists(true_file_path) and os.path.exists(predicted_path):
                real_midi_path = os.path.join(MIDI_OUTPUT_DIR, output_name + '_true.mid')
                predicted_midi_path = os.path.join(MIDI_OUTPUT_DIR, output_name + '_predicted.mid')

                abc2midi_command = str(ABCMIDI_BIN) if ABCMIDI_BIN.exists() else 'abc2midi'
                subprocess.run([abc2midi_command, true_file_path, '-o', real_midi_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                subprocess.run([abc2midi_command, predicted_path, '-o', predicted_midi_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                success_count += 1
            else:
                error_count += 1
        except Exception as e:
            logger.error(f"Failed to convert {output_name}: {e}")
            error_count += 1

    logger.info(f"✅ MIDI generation complete: {success_count} succeeded, {error_count} failed")


def step4_compute_metrics():
    """Step 4: compute MV2H metrics."""
    logger.info("=" * 60)
    logger.info("Step 4: computing MV2H metrics...")
    logger.info("=" * 60)
    compute_metrics_from_midi_dirs(
        midi_dir=MIDI_OUTPUT_DIR,
        txt_dir=TXT_OUTPUT_DIR,
        results_dir=TEST_RESULTS_DIR,
    )


def main():
    """Main entry point: run the full test pipeline."""
    import argparse

    parser = argparse.ArgumentParser(description='ListenerT5 full test pipeline')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use')
    parser.add_argument('--workers', type=int, default=NUM_WORKERS, help='Number of worker processes')
    parser.add_argument('--skip-abc', action='store_true', help='Skip ABC generation')
    parser.add_argument('--skip-midi', action='store_true', help='Skip MIDI generation')
    parser.add_argument('--only-metrics', action='store_true', help='Only compute metrics')

    args = parser.parse_args()

    mp.set_start_method('spawn', force=True)

    logger.info("🚀 Starting the ListenerT5 full test pipeline")
    logger.info(f"Configuration: GPU={args.gpu}, Workers={args.workers}")

    try:
        if args.only_metrics:
            step4_compute_metrics()
        else:
            if not args.skip_abc:
                step1_generate_abc(args.gpu, args.workers)
                step2_add_x1_header()

            if not args.skip_midi:
                step2_add_x1_header()
                step3_build_midis()

            step4_compute_metrics()

        logger.info("=" * 60)
        logger.info("🎊 All steps completed!")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()