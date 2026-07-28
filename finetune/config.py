from pathlib import Path


# To controll the size of the Neural Net
HIDDIEN_DIM = 512

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DATA_ROOT = PROJECT_ROOT / "data"
OUTPUT_ROOT = BASE_DIR / "outputs"
RUN_DIR = OUTPUT_ROOT / "Piano_Only_Render_Data_char3_patch_9_epoch25"


PATCH_STREAM = False

PATCH_LENGTH = 2048
PATCH_SIZE = 16

PATCH_NUM_LAYERS = 9    
CHAR_NUM_LAYERS = 3   

NUM_EPOCHS = 32
LEARNING_RATE = 1e-5
ACCUMULATION_STEPS = 1
BATCH_SIZE = 6
PATCH_SAMPLING_BATCH_SIZE = 0
LOAD_FROM_CHECKPOINT = True
LOAD_FROM_PRETRAINED = True
SHARE_WEIGHTS = False

TRAIN_DATA_PATH = DATA_ROOT / "partitions" / "finetune" / "train_filtered.jsonl"
VALIDATION_DATA_PATH = DATA_ROOT / "partitions" / "finetune" / "test_filtered.jsonl"
TEST_DATA_PATH = DATA_ROOT / "partitions" / "finetune" / "test_filtered.jsonl"

TEST_OUTPUT_DIR = RUN_DIR / "finetune_outputs"
MIDI_OUTPUT_DIR = RUN_DIR / "finetune_midi_outputs"
TXT_OUTPUT_DIR = RUN_DIR / "finetune_txt_outputs"
TEST_RESULTS_DIR = RUN_DIR / "finetune_results"

PRETRAINED_PATH = PROJECT_ROOT / "pretrain" / "outputs" / "Quartet_Only_Render_Data_char3_patch_9" / "pretrain_61M_best.pth"
WEIGHTS_PATH = RUN_DIR / "finetune_61M.pth"
LOGS_PATH = RUN_DIR / "a2s_finetune.txt"
TEST_WEIGHTS_PATH = RUN_DIR / "finetune_61M.pth"


def resolve_path(path_value):
	path = Path(path_value)
	return path if path.is_absolute() else PROJECT_ROOT / path