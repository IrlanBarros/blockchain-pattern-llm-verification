"""Lossless internal JSON-key and pattern-ID encoding; public CSV schema unchanged."""
from __future__ import annotations
from copy import deepcopy

from .retrieval import CompactCatalog

S1 = {'issue_summary':'s','issue_activity_type':'t','issue_challenge_categories':'h',
      'context_status':'x','candidates':'c'}
CANDIDATE = {'pattern':'p','evidence_text':'e','evidence_location':'l','rationale':'r','confidence':'f'}
S2 = {'verdict':'v','mechanism_match':'m','scope_match':'s','focus_match':'f','evidence_text':'e',
      'evidence_location':'l','justification':'j','adoption_status':'d','false_friend_detected':'b',
      'pattern_challenge_categories':'h','confidence':'c','alternative_pattern':'a','overlap_with':'o'}


def wire_schema(schema, *, stage):
    schema = deepcopy(schema)
    mapping = S1 if stage == 1 else S2
    def rename(node, names):
        node['properties'] = {names[k]: v for k,v in node['properties'].items()}
        node['required'] = [names[k] for k in node['required']]
    if stage == 1:
        rename(schema['properties']['candidates']['items'], CANDIDATE)
    rename(schema, mapping)
    return schema


def wire_legend(stage):
    mapping = S1 if stage == 1 else S2
    text = 'JSON keys: ' + ', '.join(f'{v}={k}' for k,v in mapping.items()) + '.'
    if stage == 1:
        text += ' Candidate keys: ' + ', '.join(f'{v}={k}' for k,v in CANDIDATE.items()) + '.'
    return text + ' Use supplied pattern IDs for all pattern fields. Keep every required field.'


def decode_payload(payload, catalog, *, stage, allowed=None):
    compact = CompactCatalog(catalog)
    mapping = S1 if stage == 1 else S2
    # Legacy payloads remain accepted for frozen outputs and backend compatibility.
    if not any(k in payload for k in mapping.values()):
        decoded = deepcopy(payload)
    else:
        if set(payload) != set(mapping.values()):
            raise ValueError('Compact response contains missing, mixed or unknown JSON fields')
        decoded = {k: payload[v] for k,v in mapping.items()}
        if stage == 1:
            if not isinstance(decoded['candidates'], list):
                raise ValueError('Compact candidates must be a list')
            decoded['candidates'] = [
                _decode_candidate(c) for c in decoded['candidates']]
    if stage == 1:
        for candidate in decoded.get('candidates', []):
            candidate['pattern'] = compact.canonicalize(candidate['pattern'])
            if allowed is not None and candidate['pattern'] not in allowed:
                raise ValueError('Stage 1 returned a pattern outside the persisted shortlist')
    else:
        for field in ('mechanism_match','scope_match','focus_match'):
            if field in decoded and not isinstance(decoded[field], bool):
                raise ValueError(f'{field} must be boolean')
        if decoded.get('alternative_pattern'):
            decoded['alternative_pattern'] = compact.canonicalize(decoded['alternative_pattern'])
        if not isinstance(decoded.get('overlap_with', []), list):
            raise ValueError('overlap_with must be a list')
        decoded['overlap_with'] = [compact.canonicalize(p) for p in decoded.get('overlap_with', [])]
    return decoded


def _decode_candidate(candidate):
    if not isinstance(candidate, dict) or set(candidate) != set(CANDIDATE.values()):
        raise ValueError('Compact candidate must include evidence, location, rationale and confidence')
    return {k: candidate[v] for k,v in CANDIDATE.items()}
