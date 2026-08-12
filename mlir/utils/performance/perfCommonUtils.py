import enum
import re

# Key name of the Split-K factor within serialized perf configs.
SPLITK_KEY = "splitKFactor"


def parse_perfconfig(perfconfig):
    """Parse a canonical ``prefix:key=value,...`` perfconfig string.

    Returns ``(prefix, params_dict)`` with integer values. Raises
    ``ValueError`` on any other shape (e.g. the legacy positional
    ``prefix:vN:`` form, which the tooling no longer emits). Mirrors
    ``GemmParamsAttr::getPerfConfigStr`` in ``RockAttrDefs.td``.
    """
    prefix, sep, body = perfconfig.partition(":")
    if not sep:
        raise ValueError(f"Invalid perfconfig format: {perfconfig}")
    params = {}
    for piece in body.split(","):
        piece = piece.strip()
        if not piece:
            continue
        key, eq, value = piece.partition("=")
        key = key.strip()
        if not eq or not key:
            raise ValueError(f"Invalid perfconfig format: {perfconfig}")
        params[key] = int(value.strip())
    return prefix, params


def serialize_perfconfig(prefix, params):
    """Serialize ``(prefix, params_dict)`` back to ``prefix:key=value,...``.

    Uses the compact, space-free form emitted by
    ``GemmParamsAttr::getPerfConfigStr`` so the result round-trips through
    ``rocmlir-gen --perf_config``.
    """
    body = ",".join(f"{key}={value}" for key, value in params.items())
    return f"{prefix}:{body}"


class Operation(enum.IntEnum):
    CONV = 1
    GEMM = 2
    FUSION = 3
    ATTENTION = 4
    GEMM_GEMM = 5
    CONV_GEMM = 6

    @staticmethod
    def from_name(name: str) -> "Operation":
        name = name.lower()
        if name == 'conv':
            return Operation.CONV
        elif name == 'gemm':
            return Operation.GEMM
        elif name == 'attention':
            return Operation.ATTENTION
        elif name == 'gemm_gemm':
            return Operation.GEMM_GEMM
        elif name == 'conv_gemm':
            return Operation.CONV_GEMM
        elif name == 'fusion':
            return Operation.FUSION
        else:
            raise ValueError(f"Unknown operation type {name}")


CORRECT_RESULT_RE = re.compile(r'\[1\s*1\s*1\]')


class GEMMLibrary(enum.IntEnum):
    CK = 1
    HIPBLASLT = 2

    @staticmethod
    def from_name(name: str) -> "GEMMLibrary":
        name = name.lower()
        if name == 'ck':
            return GEMMLibrary.CK
        elif name == 'hipblaslt':
            return GEMMLibrary.HIPBLASLT
        else:
            raise ValueError(f"Unknown library {name}")
