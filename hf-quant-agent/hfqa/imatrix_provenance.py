"""Validate experimental imatrix provenance before a future streaming release.

This is a receipt check, not an attestation: the worker and its pinned runtime
must still be authenticated by the existing release gate.  No release route
uses this helper yet.
"""
import re

HEX64 = re.compile(r'[0-9a-f]{64}')
HEX40 = re.compile(r'[0-9a-f]{40}')


def validate_matrix_provenance(evidence, *, source, calibration, runtime_revision):
    """Accept a matrix only for the exact source, dataset and runtime requested.

    Proxy calibration is deliberately not accepted: its provenance and quality
    require a separately qualified policy before it can be released.
    """
    if not all(isinstance(x, dict) for x in (evidence, source, calibration)):
        raise ValueError('matrix provenance has invalid types')
    if not isinstance(runtime_revision, str) or not HEX40.fullmatch(runtime_revision):
        raise ValueError('runtime revision must be a pinned commit')
    for value in (source.get('sha256'), calibration.get('sha256'), evidence.get('matrix_sha256'), evidence.get('tool_sha256')):
        if not isinstance(value, str) or not HEX64.fullmatch(value):
            raise ValueError('matrix provenance requires SHA-256 hashes')
    if evidence.get('source_sha256') != source['sha256'] or evidence.get('source_sha256_verified') is not True:
        raise ValueError('matrix was not calibrated from the pinned original source')
    if evidence.get('dataset_sha256') != calibration['sha256'] or evidence.get('dataset_id') != calibration.get('id'):
        raise ValueError('matrix calibration text differs from the request')
    if not isinstance(calibration.get('id'), str) or not calibration['id'] or evidence.get('context') != calibration.get('context'):
        raise ValueError('matrix calibration settings differ from the request')
    if type(calibration.get('context')) is not int or calibration['context'] < 1:
        raise ValueError('invalid matrix context')
    if evidence.get('runtime_revision') != runtime_revision:
        raise ValueError('matrix runtime differs from the request')
    if any(type(evidence.get(key)) is not int or evidence[key] < 1 for key in ('entries', 'chunks_processed')):
        raise ValueError('matrix contains no verified calibration data')
    # The result is what a receipt carries and what the Oracle gate checks again, so it must pass this very function a second time:
    # `source_sha256_verified` is part of it (it was verified above, otherwise nothing is returned).
    checked = {**{key: evidence[key] for key in ('source_sha256', 'matrix_sha256', 'dataset_sha256', 'dataset_id',
                                                 'context', 'runtime_revision', 'tool_sha256', 'entries', 'chunks_processed')},
               'source_sha256_verified': True}
    if evidence.get('calibration_basis') is not None:
        checked['calibration_basis'] = validate_basis(evidence['calibration_basis'])
    return checked


def validate_basis(basis):
    """Which files the matrix was calibrated on: the pinned source itself, or the Q8_0 set of the same repository revision (with the same tensors)."""
    if not isinstance(basis, dict) or basis.get('kind') not in {'source', 'q8_0_sibling'}:
        raise ValueError('invalid matrix calibration basis')
    if basis['kind'] == 'q8_0_sibling':
        files = basis.get('files')
        if (not isinstance(files, list) or not 1 <= len(files) <= 1024 or not HEX64.fullmatch(str(basis.get('tensor_name_set_sha256', '')))
                or any(not isinstance(item, dict) or not HEX64.fullmatch(str(item.get('sha256', ''))) or type(item.get('size')) is not int for item in files)):
            raise ValueError('the Q8_0 calibration basis needs its files with hashes and the tensor-name-set hash')
        return {'kind': 'q8_0_sibling', 'tensor_name_set_sha256': basis['tensor_name_set_sha256'],
                'files': [{key: item.get(key) for key in ('filename', 'size', 'sha256')} for item in files]}
    return {'kind': 'source'}
