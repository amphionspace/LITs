"""Strict, auditable FM -> iMF weights-only initialization."""

import math

from lits.models.components.improved_mean_flow import IMF_Causal


def checkpoint_statistics(checkpoint):
    stats = {name: float(checkpoint['state_dict'][name]) for name in ('mel_mean', 'mel_std')}
    if not all(math.isfinite(v) for v in stats.values()) or stats['mel_std'] <= 0:
        raise ValueError('Checkpoint Mel statistics must be finite with positive standard deviation')
    return stats


def load_imf_initial_weights(model, checkpoint, reset_speaker_embeddings=False):
    if not isinstance(model.decoder, IMF_Causal):
        raise TypeError('FM -> iMF loading requires model/cfm=imf')
    state = dict(checkpoint['state_dict'])
    prefix = 'decoder.estimator.'
    estimator_state = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    source_is_imf = any(k.startswith(('interval_projector.', 'v_up_blocks.', 'v_final_block.', 'v_final_proj.'))
                        for k in estimator_state)
    if source_is_imf:
        initialized_keys = []
    else:
        expanded = model.decoder.expand_estimator_state(estimator_state)
        initialized_keys = sorted(set(expanded) - set(estimator_state))
        state.update({prefix + k: v for k, v in expanded.items()})
    if reset_speaker_embeddings:
        if 'spk_emb.weight' not in state or not hasattr(model, 'spk_emb'):
            raise ValueError('Speaker reset requested for a checkpoint/model without speaker embeddings')
        state['spk_emb.weight'] = model.spk_emb.weight.detach().clone()
    model.load_state_dict(state, strict=True)
    return dict(source_objective='imf' if source_is_imf else 'fm', target_objective='imf',
                source_step=checkpoint.get('global_step'), new_estimator_keys=initialized_keys,
                speaker_embeddings_reset=reset_speaker_embeddings, optimizer_loaded=False,
                scheduler_loaded=False)
