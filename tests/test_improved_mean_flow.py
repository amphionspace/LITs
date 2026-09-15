"""iMF objective, FM migration, two-step sampling, and real decoder gradients."""

import pytest
import torch
from torch import nn
from omegaconf import OmegaConf

from lits.models.components.flow_matching import CFM_Causal
from lits.models.components.improved_mean_flow import IMF_Causal, imf_loss, sample_tr
from lits.utils.imf_checkpoint import checkpoint_statistics, load_imf_initial_weights


def flow_config(**overrides):
    return OmegaConf.create(dict(name='IMF', solver='euler', sigma_min=0.0,
        num_steps=2, sampling_time_grid=[0.0, 0.5, 1.0], **overrides))


def tiny_flow(cls=IMF_Causal):
    return cls(in_channels=16, out_channel=8, cfm_params=flow_config(),
        decoder_params=dict(channels=[16, 16], dropout=0.0, attention_head_dim=8,
                            n_blocks=1, num_mid_blocks=1, num_heads=1, act_fn='snakebeta',
                            static_chunk_size=4, num_decoding_left_chunks=-1, decoder_left_frames=4),
        n_spks=2, spk_emb_dim=4)


class TinyModel(nn.Module):
    def __init__(self, cls=IMF_Causal):
        super().__init__()
        self.spk_emb = nn.Embedding(2, 4)
        self.encoder = nn.Linear(8, 8)
        self.decoder = tiny_flow(cls)


@pytest.mark.parametrize('diagonal', [False, True])
def test_objective_and_gradients_match_analytic_upstream_identity(diagonal):
    # Analytic field: u=a*z+b*t+c*r, v=d*z+e*t. Unlike a duplicated JVP
    # implementation, the reference differentiates this field by hand.
    parameters = [torch.tensor(v, requires_grad=True) for v in (.2, .3, .4, -.1, .7)]
    a, b, c, d, e = parameters
    x = torch.tensor([[[.1, -.3, .8]], [[.7, .2, 0.]]])
    noise = torch.tensor([[[.3, .2, -.5]], [[-.2, .5, 0.]]])
    mask = torch.tensor([[[1., 1., 1.]], [[1., 1., 0.]]])
    t = torch.tensor([.8, .6])[:, None, None]
    r = t if diagonal else torch.tensor([.2, .1])[:, None, None]

    def field(z, t, r):
        return (a*z + b*t + c*r)*mask, (d*z + e*t)*mask

    actual, _, stats = imf_loss(field, x, mask, t, r, noise)
    z = ((1-t)*x + t*noise)*mask
    predicted_tangent = (d*z + e*t)*mask
    derivative = ((a*predicted_tangent + b)*mask).detach()
    target = (noise-x)*mask
    u, v = field(z, t, r)
    err_u = ((u+(t-r)*derivative-target).square()*mask).sum((1, 2))
    err_v = ((v-target).square()*mask).sum((1, 2))
    expected = (err_u/(err_u+.01).detach() + err_v/(err_v+.01).detach()).mean()
    torch.testing.assert_close(actual, expected)
    actual_grads = torch.autograd.grad(actual, parameters, retain_graph=True)
    expected_grads = torch.autograd.grad(expected, parameters)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad)
    assert stats['fm_fraction'] == float(diagonal)


def test_padding_does_not_change_objective_or_parameter_gradient():
    results = []
    for extra in [0, 7]:
        weight = torch.tensor(.3, requires_grad=True)
        x = torch.ones(2, 3, 5+extra)
        mask = (torch.arange(5+extra)[None, None, :] < torch.tensor([3, 5])[:, None, None]).float()
        t = torch.full((2, 1, 1), .8)
        r = torch.full_like(t, .2)
        def field(z, t, r):
            return weight*(z+t+r)*mask, weight*(z+t)*mask
        loss, _, _ = imf_loss(field, x, mask, t, r, x*.2)
        results.append((loss.detach(), torch.autograd.grad(loss, weight)[0]))
    torch.testing.assert_close(results[0][0], results[1][0])
    torch.testing.assert_close(results[0][1], results[1][1])


def test_sampler_uses_ordered_logit_normals_and_upstream_fm_proportion():
    t, r, diagonal = sample_tr(1000, 'cpu')
    assert torch.all((0 < r) & (r <= t) & (t < 1))
    assert diagonal.sum() == 500
    assert torch.equal(t[:500], r[:500])
    assert torch.all(t[500:] > r[500:])


def test_fm_migration_preserves_old_field_copies_v_and_rejects_missing_weights():
    torch.manual_seed(3)
    fm = TinyModel(CFM_Causal).eval()
    imf = TinyModel().eval()
    checkpoint = {'state_dict': fm.state_dict(), 'global_step': 21000}
    before = imf.spk_emb.weight.detach().clone()
    report = load_imf_initial_weights(imf, checkpoint, reset_speaker_embeddings=True)
    assert report['source_objective'] == 'fm' and report['source_step'] == 21000
    torch.testing.assert_close(before, imf.spk_emb.weight, rtol=0, atol=0)
    for name, tensor in fm.state_dict().items():
        if name != 'spk_emb.weight':
            torch.testing.assert_close(imf.state_dict()[name], tensor, rtol=0, atol=0)
    for name, tensor in imf.decoder.estimator.v_up_blocks.state_dict().items():
        torch.testing.assert_close(tensor, fm.decoder.estimator.up_blocks.state_dict()[name], rtol=0, atol=0)
    x, mu, mask, spks = torch.randn(2, 8, 16), torch.randn(2, 8, 16), torch.ones(2, 1, 16), torch.randn(2, 4)
    t = torch.tensor([.2, .7])
    with torch.no_grad():
        original = fm.decoder.estimator(x, mask, mu, 1-t, spks)
        u, v = imf.decoder.estimator.forward_uv(x, mask, mu, t[:, None, None], t[:, None, None], spks)
    torch.testing.assert_close(-u, original)
    torch.testing.assert_close(-v, original)
    broken = dict(checkpoint['state_dict'])
    del broken['decoder.estimator.final_proj.weight']
    with pytest.raises(ValueError, match='keys differ'):
        load_imf_initial_weights(imf, {'state_dict': broken})
    broken = dict(checkpoint['state_dict'])
    del broken['encoder.weight']
    with pytest.raises(RuntimeError, match='Missing key'):
        load_imf_initial_weights(imf, {'state_dict': broken})


def test_imf_checkpoint_reload_preserves_auxiliary_head_and_interval():
    original, restored = TinyModel(), TinyModel()
    with torch.no_grad():
        original.decoder.estimator.interval_projector.linear_2.weight.fill_(.3)
        original.decoder.estimator.v_final_proj.weight.fill_(.7)
    report = load_imf_initial_weights(restored, {'state_dict': original.state_dict()})
    assert report['source_objective'] == 'imf' and not report['new_estimator_keys']
    for name, tensor in original.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], tensor, rtol=0, atol=0)


def test_two_step_solver_uses_configured_intervals_and_only_u():
    flow = tiny_flow().eval()
    class IntervalField(nn.Module):
        interval_projector = True
        def __init__(self):
            super().__init__()
            self.calls = []
        def forward(self, x, mask, mu, t, spks, cond, streaming, r):
            self.calls.append((float(r), float(t)))
            return x + (t-r)
    field = IntervalField()
    flow.estimator = field
    x = torch.ones(1, 8, 8)
    out = flow.solve_euler(x, torch.linspace(0, 1, 3), x, torch.ones(1, 1, 8), None, None, False)
    expected = x + .5*(x+.5)
    expected = expected + .5*(expected+.5)
    torch.testing.assert_close(out, expected)
    assert field.calls == [(0., .5), (.5, 1.)]


def test_streaming_sampler_matches_explicit_upstream_updates_and_skips_v_tail():
    flow = tiny_flow().eval()
    x, mu, mask, spks = torch.randn(1, 8, 16), torch.randn(1, 8, 16), torch.ones(1, 1, 16), torch.randn(1, 4)
    def fail_v(*args):
        pytest.fail('Auxiliary v tail executed during inference')
    handle = flow.estimator.v_final_proj.register_forward_pre_hook(fail_v)
    with torch.inference_mode():
        expected = x.clone()
        for start, end in [(0., .5), (.5, 1.)]:
            velocity, _ = flow.estimator.forward_streaming(
                expected, mask, mu, torch.tensor(end), spks, r=torch.tensor(start))
            expected = expected + (end-start)*velocity
        actual = flow.solve_euler(x, torch.linspace(0, 1, 3), mu, mask, spks, None, True)
        torch.testing.assert_close(actual, expected)
        assert len(flow._decoder_caches) == 2
        assert flow._decoder_caches[0] is not flow._decoder_caches[1]
        prefix_noise = torch.cat([x, torch.randn_like(x)], dim=-1)
        prefix_mu = torch.cat([mu, torch.randn_like(mu)], dim=-1)
        continued = flow.solve_euler(prefix_noise, torch.linspace(0, 1, 3), prefix_mu,
                                    torch.ones(1, 1, 32), spks, None, True, chunk_start=16)
        assert continued.shape[-1] == 32 and torch.isfinite(continued).all()
        with pytest.raises(ValueError, match='time grid'):
            flow.solve_euler(prefix_noise, torch.linspace(0, 1, 5), prefix_mu,
                             torch.ones(1, 1, 32), spks, None, True, chunk_start=16)
    handle.remove()


def test_checkpoint_normalization_is_validated():
    assert checkpoint_statistics({'state_dict': {'mel_mean': torch.tensor(-5.5), 'mel_std': torch.tensor(3.)}}) == {
        'mel_mean': -5.5, 'mel_std': 3.}
    for mean, std in [(float('nan'), 1.), (0., 0.), (0., -1.)]:
        with pytest.raises(ValueError, match='statistics'):
            checkpoint_statistics({'state_dict': {'mel_mean': mean, 'mel_std': std}})


@pytest.mark.parametrize('precision', ['float32', 'bf16'])
def test_real_decoder_jvp_backward_and_inference_mode_validation(precision):
    flow = tiny_flow().train()
    x = torch.randn(2, 8, 16)
    mu = torch.randn_like(x, requires_grad=True)
    spks = torch.randn(2, 4, requires_grad=True)
    mask = (torch.arange(16)[None, None, :] < torch.tensor([12, 16])[:, None, None]).float()
    with torch.autocast('cpu', dtype=torch.bfloat16, enabled=precision == 'bf16'):
        loss, _ = flow.compute_loss(x, mask, mu, spks)
    assert torch.isfinite(loss)
    loss.backward()
    assert mu.grad is not None and mu.grad.abs().sum() > 0
    assert spks.grad is not None and spks.grad.abs().sum() > 0
    for name in ['final_proj.weight', 'v_final_proj.weight', 'interval_projector.linear_2.weight']:
        grad = dict(flow.estimator.named_parameters())[name].grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0, name
    flow.eval()
    with torch.inference_mode():
        val_loss, _ = flow.compute_loss(x.clone(), mask.clone(), mu.detach().clone(), spks.detach().clone())
    assert torch.isfinite(val_loss)


def test_fm_and_imf_hydra_configs_are_independent(monkeypatch):
    from pathlib import Path
    from hydra import compose, initialize_config_dir
    root = Path(__file__).resolve().parents[1]
    for key, value in dict(PROJECT_ROOT=str(root), TRAIN_FILELIST='/tmp/train.txt',
                           VALID_FILELIST='/tmp/val.txt', STAGE2_DATA='/tmp/data', N_SPKS='2').items():
        monkeypatch.setenv(key, value)
    with initialize_config_dir(config_dir=str(root/'configs'), version_base='1.3'):
        fm = compose(config_name='train', overrides=['experiment=en-zh'])
        imf = compose(config_name='train', overrides=['experiment=en-zh-imf-stage2'])
    assert fm.model.cfm.name == 'CFM' and fm.model.cfm.sigma_min == 1e-4
    assert imf.model.cfm.name == 'IMF' and imf.model.cfm.sigma_min == 0
    assert imf.model.cfm.num_steps == 2
    assert list(imf.model.cfm.sampling_time_grid) == [0., .5, 1.]
    assert imf.data.data_dir == '/tmp/data' and not imf.data.load_durations
    assert imf.init_reset_speaker_embeddings and not fm.init_reset_speaker_embeddings
