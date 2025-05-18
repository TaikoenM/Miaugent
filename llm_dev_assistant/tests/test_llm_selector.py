# tests/test_llm_selector.py
import os
import unittest
from unittest.mock import MagicMock, patch

from llm_dev_assistant.llm.llm_selector import (
    TaskPurpose,
    TaskComplexity,
    ModelType,
    LLMSelector,
    LLMFactory,
    choose_llm
)
from llm_dev_assistant.llm.llm_interface import LLMInterface


class TestLLMSelector(unittest.TestCase):
    """Test the LLM selector functionality."""

    @patch('llm_dev_assistant.llm.llm_selector.LLMFactory.create_vision_llm')
    def test_select_vision_llm(self, mock_create_vision):
        """Test selecting a vision LLM for vision tasks."""
        # Create a mock LLM instance
        mock_llm = MagicMock(spec=LLMInterface)
        mock_create_vision.return_value = mock_llm

        selector = LLMSelector()
        llm, model_type = selector.select_llm(
            task_purpose=TaskPurpose.VISION,
            task_complexity=TaskComplexity.NORMAL,
            vision_model_name="test-vision-model"
        )

        # Verify that the correct model was selected
        self.assertEqual(model_type, "VISION")
        self.assertEqual(llm, mock_llm)
        mock_create_vision.assert_called_once_with("test-vision-model")

    @patch('llm_dev_assistant.llm.llm_selector.LLMFactory.create_online_llm')
    def test_select_online_llm_for_hard_task(self, mock_create_online):
        """Test selecting an online LLM for hard tasks."""
        # Create a mock LLM instance
        mock_llm = MagicMock(spec=LLMInterface)
        mock_create_online.return_value = mock_llm

        selector = LLMSelector()
        llm, model_type = selector.select_llm(
            task_purpose=TaskPurpose.CODE_GENERATION,
            task_complexity=TaskComplexity.HARD,
            online_model_name="test-online-model"
        )

        # Verify that the correct model was selected
        self.assertEqual(model_type, "ONLINE")
        self.assertEqual(llm, mock_llm)
        mock_create_online.assert_called_once_with("test-online-model")

    @patch('llm_dev_assistant.llm.llm_selector.LLMFactory.create_local_llm')
    def test_select_local_llm_for_normal_task(self, mock_create_local):
        """Test selecting a local LLM for normal tasks."""
        # Create a mock LLM instance
        mock_llm = MagicMock(spec=LLMInterface)
        mock_create_local.return_value = mock_llm

        selector = LLMSelector()
        llm, model_type = selector.select_llm(
            task_purpose=TaskPurpose.TEXT_GENERATION,
            task_complexity=TaskComplexity.NORMAL,
            local_model_path="/path/to/local/model"
        )

        # Verify that the correct model was selected
        self.assertEqual(model_type, "LOCAL")
        self.assertEqual(llm, mock_llm)
        mock_create_local.assert_called_once_with("/path/to/local/model")

    @patch('llm_dev_assistant.llm.llm_selector.LLMFactory.create_lmstudio_llm')
    def test_select_lmstudio_llm(self, mock_create_lmstudio):
        """Test selecting an LM Studio LLM."""
        # Create a mock LLM instance
        mock_llm = MagicMock(spec=LLMInterface)
        mock_create_lmstudio.return_value = mock_llm

        selector = LLMSelector()
        llm, model_type = selector.select_llm(
            task_purpose=TaskPurpose.CODE_DEBUGGING,
            task_complexity=TaskComplexity.LOW,
            lm_studio_model="llama2-7b"
        )

        # Verify that the correct model was selected
        self.assertEqual(model_type, "LMSTUDIO")
        self.assertEqual(llm, mock_llm)
        mock_create_lmstudio.assert_called_once_with("llama2-7b")

    @patch('llm_dev_assistant.llm.llm_selector.LLMFactory.create_online_llm')
    def test_fallback_to_online_llm(self, mock_create_online):
        """Test falling back to online LLM when no local or LM Studio model specified."""
        # Create a mock LLM instance
        mock_llm = MagicMock(spec=LLMInterface)
        mock_create_online.return_value = mock_llm

        selector = LLMSelector()
        llm, model_type = selector.select_llm(
            task_purpose=TaskPurpose.TEXT_SUMMARIZATION,
            task_complexity=TaskComplexity.NORMAL,
            online_model_name="test-fallback-model"
        )

        # Verify that it fell back to the online model
        self.assertEqual(model_type, "ONLINE")
        self.assertEqual(llm, mock_llm)
        mock_create_online.assert_called_once_with("test-fallback-model")

    @patch('llm_dev_assistant.llm.llm_selector.LLMSelector')
    def test_choose_llm_helper_function(self, mock_selector_class):
        """Test the choose_llm helper function."""
        # Create a mock for the LLMSelector instance
        mock_selector = MagicMock()
        mock_selector_class.return_value = mock_selector

        # Set up the return value for select_llm
        mock_llm = MagicMock(spec=LLMInterface)
        mock_selector.select_llm.return_value = (mock_llm, "TEST")

        # Call the choose_llm function
        result = choose_llm(
            task_purpose="CODE_GENERATION",
            task_complexity="NORMAL",
            online_model="test-model",
            local_model="/test/path"
        )

        # Verify the result and that select_llm was called correctly
        self.assertEqual(result, mock_llm)
        mock_selector.select_llm.assert_called_once_with(
            task_purpose=TaskPurpose.CODE_GENERATION,
            task_complexity=TaskComplexity.NORMAL,
            vision_model_name="gpt-4-vision-preview",
            online_model_name="test-model",
            local_model_path="/test/path",
            lm_studio_model=None
        )

    def test_llm_factory_methods(self):
        """Test that LLM factory methods work correctly."""
        # Mock the imports to avoid actual implementation
        with patch('llm_dev_assistant.llm.llm_selector.VisionLLMAdapter') as mock_vision, \
                patch('llm_dev_assistant.llm.llm_selector.OpenAIAdapter') as mock_openai, \
                patch('llm_dev_assistant.llm.llm_selector.LocalLLMAdapter') as mock_local, \
                patch('llm_dev_assistant.llm.llm_selector.LMStudioAdapter') as mock_lmstudio:
            # Set up the mocks to return a mock LLM
            mock_vision.return_value = MagicMock(spec=LLMInterface)
            mock_openai.return_value = MagicMock(spec=LLMInterface)
            mock_local.return_value = MagicMock(spec=LLMInterface)
            mock_lmstudio.return_value = MagicMock(spec=LLMInterface)

            # Test each factory method
            vision_llm = LLMFactory.create_vision_llm("vision-model")
            online_llm = LLMFactory.create_online_llm("online-model")
            local_llm = LLMFactory.create_local_llm("/path/to/model")
            lmstudio_llm = LLMFactory.create_lmstudio_llm("studio-model")

            # Verify that the correct constructors were called
            mock_vision.assert_called_once_with(model="vision-model")
            mock_openai.assert_called_once_with(model="online-model")
            mock_local.assert_called_once_with(model_path="/path/to/model")
            mock_lmstudio.assert_called_once_with(model_name="studio-model")

            # Verify that the factory methods returned the mock LLMs
            self.assertEqual(vision_llm, mock_vision.return_value)
            self.assertEqual(online_llm, mock_openai.return_value)
            self.assertEqual(local_llm, mock_local.return_value)
            self.assertEqual(lmstudio_llm, mock_lmstudio.return_value)

    def test_cache_llm_instances(self):
        """Test that LLM instances are cached."""
        # Create a selector with mocked factory methods
        selector = LLMSelector()

        # Mock the factory methods
        selector.factories = {
            ModelType.VISION: MagicMock(return_value=MagicMock(spec=LLMInterface)),
            ModelType.ONLINE: MagicMock(return_value=MagicMock(spec=LLMInterface)),
            ModelType.LOCAL: MagicMock(return_value=MagicMock(spec=LLMInterface)),
            ModelType.LMSTUDIO: MagicMock(return_value=MagicMock(spec=LLMInterface))
        }

        # Call select_llm twice with the same parameters
        llm1, _ = selector.select_llm(
            task_purpose=TaskPurpose.CODE_GENERATION,
            task_complexity=TaskComplexity.NORMAL,
            online_model_name="test-model"
        )

        llm2, _ = selector.select_llm(
            task_purpose=TaskPurpose.CODE_GENERATION,
            task_complexity=TaskComplexity.NORMAL,
            online_model_name="test-model"
        )

        # Verify that the same instance was returned and the factory was called only once
        self.assertEqual(llm1, llm2)
        selector.factories[ModelType.ONLINE].assert_called_once()

    def test_clear_instance(self):
        """Test clearing a specific LLM instance from the cache."""
        # Create a selector with mock instances
        selector = LLMSelector()
        mock_llm = MagicMock(spec=LLMInterface)
        model_key = f"{ModelType.ONLINE.name}:model_name=test-model"
        selector.llm_instances[model_key] = mock_llm

        # Clear the instance
        result = selector.clear_instance(model_key)

        # Verify that the instance was cleared
        self.assertTrue(result)
        self.assertNotIn(model_key, selector.llm_instances)

    def test_clear_all_instances(self):
        """Test clearing all LLM instances from the cache."""
        # Create a selector with multiple instances
        selector = LLMSelector()
        selector.llm_instances = {
            f"{ModelType.ONLINE.name}:model_name=test-model": MagicMock(spec=LLMInterface),
            f"{ModelType.LOCAL.name}:model_path=/path/to/model": MagicMock(spec=LLMInterface),
            f"{ModelType.LMSTUDIO.name}:model_name=studio-model": MagicMock(spec=LLMInterface)
        }

        # Clear all instances
        selector.clear_all_instances()

        # Verify that all instances were cleared
        self.assertEqual(len(selector.llm_instances), 0)


if __name__ == '__main__':
    unittest.main()
