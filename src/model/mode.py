import enum


class Mode(enum.Enum):
    NONE = 0
    ONE_PATH_FIXED = 1
    ONE_PATH_RANDOM = 2
    TWO_PATHS = 3
    ALL_PATHS = 4
    PROJECT = 5
    NAS_BENCH = 6
    LLM_PATH = 7
