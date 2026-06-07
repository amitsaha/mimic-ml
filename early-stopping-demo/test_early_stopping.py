import unittest
from unittest.mock import MagicMock, call

from early_stopping import EarlyStopping


class TestEarlyStopping(unittest.TestCase):
    def test_early_stopping_first_call(self):
        sut = EarlyStopping()
        sut.best_loss = None
        sut.save_checkpoint = MagicMock()
        model = None
        current_loss = 0.01

        sut(current_loss, model)

        sut.save_checkpoint.assert_called_with(0.01, model)

    def test_early_stopping_call_outside_delta(self):
        sut = EarlyStopping(min_delta=0.001)
        sut.best_loss = 0.05
        sut.save_checkpoint = MagicMock()
        model = None
        current_loss = 0.049

        sut(current_loss, model)

        sut.save_checkpoint.assert_called_with(0.049, model)

    # early stopping loop
    def test_early_stopping_call_within_delta_loop(self):
        sut = EarlyStopping(min_delta=0.001)
        sut.save_checkpoint = MagicMock()
        model = None
        # loss decreases then increases (overfitting - validation loss)
        losses = [0.45, 0.40, 0.39, 0.41, 0.43, 0.45, 0.48, 0.50, 0.51, 0.52, 0.53]

        for i in range(10):  # epochs
            sut(losses[i], model)

            if sut.early_stop:
                break

        self.assertEqual(sut.counter, 5)  # default patience

        sut.save_checkpoint.assert_has_calls(
            [call(0.45, model), call(0.40, model), call(0.39, model)]
        )
