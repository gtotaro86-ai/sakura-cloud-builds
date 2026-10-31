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
    return {**{key: evidence[key] for key in ('source_sha256', 'matrix_sha256', 'dataset_sha256', 'dataset_id',
                                              'context', 'runtime_revision', 'tool_sha256', 'entries', 'chunks_processed')},
            'source_sha256_verified': True}
