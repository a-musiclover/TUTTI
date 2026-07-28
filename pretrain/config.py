from pathlib import Path


# To controll the size of the Neural Net
HIDDIEN_DIM = 512

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DATA_ROOT = PROJECT_ROOT / "data"
OUTPUT_ROOT = BASE_DIR / "outputs"
RUN_DIR = OUTPUT_ROOT / "Quartet_Only_Render_Data_char3_patch_9"


PATCH_STREAM = False

PATCH_LENGTH = 2048
PATCH_SIZE = 16

PATCH_NUM_LAYERS = 9     # The original setting for this is 9, smaller was 6
CHAR_NUM_LAYERS = 3   # The original setting for this is 3, smaller was 2

NUM_EPOCHS = 32
SAVE_EVERY = 5
LEARNING_RATE = 2e-4
ACCUMULATION_STEPS = 1
BATCH_SIZE = 10
PATCH_SAMPLING_BATCH_SIZE = 0
LOAD_FROM_CHECKPOINT = True
LOAD_FROM_PRETRAINED = True
SHARE_WEIGHTS = False

TRAIN_DATA_PATH = DATA_ROOT / "partitions" / "pretrain" / "train.jsonl"
VALIDATION_DATA_PATH = DATA_ROOT / "partitions" / "pretrain" / "valid.jsonl"
TEST_DATA_PATH = DATA_ROOT / "partitions" / "pretrain" / "test.jsonl"

TEST_OUTPUT_DIR = RUN_DIR / "test_outputs_pretrain"
MIDI_OUTPUT_DIR = RUN_DIR / "midi_outputs_pretrain"
TXT_OUTPUT_DIR = RUN_DIR / "txt_outputs_pretrain"


PRETRAINED_PATH = BASE_DIR / "pretrained_weight.pth"
WEIGHTS_PATH = RUN_DIR / "pretrain_61M.pth"
BEST_WEIGHTS_PATH = RUN_DIR / "pretrain_61M_best.pth"
LOGS_PATH = RUN_DIR / "pretrain_61M_task.txt"
TEST_WEIGHTS_PATH = RUN_DIR / "pretrain_61M_task.pth"


def resolve_path(path_value):
	path = Path(path_value)
	return path if path.is_absolute() else PROJECT_ROOT / path