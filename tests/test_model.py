import unittest
import torch
import torch.nn.functional as F
from ttse.models.states import AnchoredFastWeightState, build_state
from ttse.models.system import StreamingTSE
from ttse.frontier import configurations


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)

    def test_update_matches_gradient_step(self):
        m = AnchoredFastWeightState(dim=8)
        a, e = F.normalize(torch.randn(2, 8), dim=-1), torch.randn(2, 8)
        st = m.init(a)
        st['W'] = torch.randn(2, 8, 8, requires_grad=True)
        energy = torch.randn(2)
        k, v = m.K(e), m.V(e)
        loss = .5 * ((st['W'] @ k.unsqueeze(-1)).squeeze(-1) - v).square().sum()
        grad, = torch.autograd.grad(loss, st['W'])
        conf = F.cosine_similarity(e, a)
        gates = m.gate(torch.cat([e, conf[:, None], energy[:, None]], -1)).sigmoid()
        expected = (1 - .1 * gates[:, 1, None, None]) * st['W'] - gates[:, 0, None, None] * grad
        nxt, s = m.step(st, e, {'energy': energy})
        torch.testing.assert_close(nxt['W'], expected)
        torch.testing.assert_close(s, F.normalize(a + (expected @ m.q), dim=-1))

    def test_all_22_settings_and_oracle_gate(self):
        cfgs = configurations()
        self.assertEqual(len(cfgs), 22)
        self.assertEqual(len({c['id'] for c in cfgs}), 22)
        self.assertEqual([sum(c['state'] == n for c in cfgs) for n in ('ema','gated_ema','vad_gated')], [9,8,5])
        a, e = F.normalize(torch.randn(2,128),dim=-1), F.normalize(torch.randn(2,128),dim=-1)
        for c in cfgs:
            m = build_state(c['state'], **c['state_kw'])
            _, s = m.step(m.init(a), e, {'active':torch.zeros(2)})
            self.assertTrue(torch.isfinite(s).all())
            if c.get('oracle_activity'):
                torch.testing.assert_close(s,a)

    def test_streaming_tail_and_bptt(self):
        m = StreamingTSE(chunk_ms=4, ctx_chunks=2,
            state_kw={'dim':8}, backbone_kw=dict(n_filters=8,bn_ch=8,hid_ch=8,n_blocks=1,n_repeats=1,state_dim=8),
            spk_kw=dict(n_filters=8,ch=8,emb_dim=8,n_blocks=1))
        for part in (m.backbone,m.spk_enc):
            for p in part.parameters(): p.requires_grad=False
        mix, enroll = torch.randn(1,129),torch.randn(1,96)
        y = m.forward_streaming(mix,enroll)
        self.assertEqual(y.shape,mix.shape)
        self.assertTrue(torch.isfinite(y).all())
        y.square().mean().backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in m.state.parameters()))
        self.assertTrue(all(p.grad is None for p in m.backbone.parameters()))


if __name__ == '__main__':
    unittest.main()
