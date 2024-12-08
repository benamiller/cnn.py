import unittest
from src.mse import MSE

class TestMSE(unittest.TestCase):
    def test_get_mean_squared_error(self):
        preds = [1, 2, 3]
        labels = [2, 3, 4]
        mse = MSE(preds, labels)

        self.assertEqual(mse.get_mean_squared_error(), 1)


if __name__ == '__main__':
    unittest.main()
        
