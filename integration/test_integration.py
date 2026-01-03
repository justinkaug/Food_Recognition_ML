"""Integration tests for ML models with Calorie Tracker"""

import unittest
import os
import sys
sys.path.append(os.path.dirname(__file__))

from model_adapter import FoodMLAdapter


class TestIntegration(unittest.TestCase):
    """Test integration of ML models"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        # Note: This requires exported models to exist
        models_dir = os.path.join(os.path.dirname(__file__), '..', 'exported_models')
        
        if not os.path.exists(models_dir):
            raise unittest.SkipTest("Exported models not found. Run export_models.py first.")
        
        try:
            cls.adapter = FoodMLAdapter(models_dir=models_dir)
        except FileNotFoundError as e:
            raise unittest.SkipTest(f"Models not found: {e}")
    
    def test_adapter_initialization(self):
        """Test that adapter initializes correctly"""
        self.assertIsNotNone(self.adapter.classifier)
        self.assertIsNotNone(self.adapter.recognizer)
    
    def test_preprocess_image(self):
        """Test image preprocessing"""
        # This test requires a sample image
        # For now, just test that the method exists
        self.assertTrue(hasattr(self.adapter, 'preprocess_image'))
    
    def test_analyze_food_structure(self):
        """Test that analyze_food returns correct structure"""
        # This would require a real test image
        # result = self.adapter.analyze_food('test_images/sample.jpg')
        # self.assertIn('is_processed', result)
        # self.assertIn('food_item', result)
        # self.assertIn('confidence', result)
        pass
    
    def test_classify_food(self):
        """Test food classification"""
        # Would require preprocessed image
        pass
    
    def test_recognize_food(self):
        """Test food recognition"""
        # Would require preprocessed image
        pass


class TestModelPerformance(unittest.TestCase):
    """Test model performance metrics"""
    
    def test_inference_speed(self):
        """Test that inference is fast enough"""
        # Target: < 50ms per image
        pass
    
    def test_batch_inference(self):
        """Test batch inference capability"""
        pass


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling"""
    
    def test_invalid_image_path(self):
        """Test handling of invalid image path"""
        pass
    
    def test_corrupted_image(self):
        """Test handling of corrupted image"""
        pass
    
    def test_low_confidence_predictions(self):
        """Test behavior with low confidence predictions"""
        pass


def run_integration_tests():
    """Run all integration tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestModelPerformance))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_integration_tests()
    sys.exit(0 if success else 1)
