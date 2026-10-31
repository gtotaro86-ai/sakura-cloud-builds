"""Experimental, deliberately narrow matrix-stream qualification policy.

This checks a known successful parity+inference fixture only. It does not
authenticate a worker or authorize staging; the release gate must additionally
bind GitHub run, worker revision, output validation, inference and Hub staging.
"""
from hfqa.imatrix_data import DATASET_ID, DATASET_SHA256
from hfqa.imatrix_provenance import validate_matrix_provenance
from hfqa.stream_imatrix import EXPERIMENTAL_LLAMA_REVISION, EXPERIMENTAL_PRESETS

SOURCE = {'repo_id': 'Qwen/Qwen2.5-0.5B-Instruct-GGUF',
          'revision': '9217f5db79a29953eb74d5343926648285ec7e67',
          'filename': 'qwen2.5-0.5b-instruct-q8_0.gguf', 'size': 675710816,
          'sha256': 'ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e'}
IMATRIX_TOOL_SHA256 = 'e1fc107576eed4b3ba0ac7019caa4fbbc1f515082d8c9c9324046f3d5e6d9721'
# Binaries of the unchanged upstream commit as built by the pinned GitHub workflow (shared libraries, bridge, completion tool). The same hashes came out of
# four independent GitHub runs (40941821910, 41023286959, 41024314715, 41025178846); a runner image change would change them and is refused until re-qualified.
NATIVE_PINS = {'bridge_sha256': 'de6b70dbe9884e70cc7a32f0aa1e9e9615ce53f9135db32f517fff9c27f14cd4',
               'ggml_sha256': 'e843caf2f95799678bf0624874dcd7c3466a0584cbf62b3c93b8805a8697bb28',
               'llama_sha256': '04570f50b8e0b96714d180c9825670701a65215c2432922a2b5147285b7714b5'}
INFERENCE_SHA256 = '323e0b05a9b3e7ed30851d4a393cf457e7a4372dbc0d1be2e93bc16235721fe9'


def validate_native(native, inference):
    """The runtime that quantized and tested the output must be exactly the qualified one: same commit, same library, bridge and test-tool bytes."""
    if not isinstance(native, dict) or not isinstance(inference, dict):
        raise ValueError('native runtime identity is missing')
    if native.get('runtime_revision') != EXPERIMENTAL_LLAMA_REVISION:
        raise ValueError('runtime revision is not the qualified one')
    if any(native.get(key) != value for key, value in NATIVE_PINS.items()):
        raise ValueError('bridge or library bytes differ from the qualified build')
    if inference.get('runtime_sha256') != INFERENCE_SHA256:
        raise ValueError('the inference tool differs from the qualified build')
    if native.get('imatrix_tool_sha256') != IMATRIX_TOOL_SHA256:
        raise ValueError('imatrix binary differs from the qualified proof')


def validate_fixture(manifest, source, evidence, quant):
    """Validate only the exact source, tools and options exercised on GitHub."""
    if not isinstance(manifest, dict) or not isinstance(source, dict):
        raise ValueError('invalid stream qualification input')
    if any(source.get(k) != v for k, v in SOURCE.items()):
        raise ValueError('source is not the qualified Qwen fixture')
    if manifest.get('runtime_revision') != EXPERIMENTAL_LLAMA_REVISION or quant not in EXPERIMENTAL_PRESETS:
        raise ValueError('runtime or quant is not qualified by this proof')
    spec = manifest.get('imatrix') or {}
    if (not isinstance(spec, dict) or spec.get('method') != 'stream_imatrix' or spec.get('provenance_v1') is not True
            or spec.get('dataset_id') != DATASET_ID or spec.get('dataset_sha256') != DATASET_SHA256 or spec.get('context') != 128):
        raise ValueError('matrix settings differ from the qualified fixture')
    checked = validate_matrix_provenance(evidence, source=source,
        calibration={'id': DATASET_ID, 'sha256': DATASET_SHA256, 'context': 128}, runtime_revision=EXPERIMENTAL_LLAMA_REVISION)
    if checked['tool_sha256'] != IMATRIX_TOOL_SHA256:
        raise ValueError('imatrix binary differs from the qualified proof')
    return checked
