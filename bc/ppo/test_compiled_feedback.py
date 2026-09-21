import unittest
import torch
from exact_model import ActionHead
from ppo.compiled_feedback import compiled_feedback


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class CompiledFeedbackTest(unittest.TestCase):
    @torch.no_grad()
    def test_matches_gru_and_reads_changed_weights(self):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.manual_seed(91)
        for ledger_width, context_width in ((128, 192), (112, 256)):
            head = ActionHead(128, ledger_width, context_width).cuda().eval()
            run = compiled_feedback(head)
            prefix = torch.randn(5, 256, device='cuda')
            chosen = torch.randn(5, 256, device='cuda')
            quantity = torch.tensor([-3, 0, 1, 10, 100], device='cuda')
            delta = torch.randn(5, ledger_width, device='cuda')
            original = run(prefix, chosen, quantity, delta).clone()
            torch.testing.assert_close(original, head.advance(prefix, chosen, quantity, delta), atol=2e-6, rtol=2e-5)
            head.feedback.bias_ih.add_(0.1)
            changed = run(prefix, chosen, quantity, delta)
            torch.testing.assert_close(changed, head.advance(prefix, chosen, quantity, delta), atol=2e-6, rtol=2e-5)
            self.assertGreater((changed-original).abs().max().item(), 1e-4)


if __name__ == '__main__':
    unittest.main()


