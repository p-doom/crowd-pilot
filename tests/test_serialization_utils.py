import unittest
from unittest.mock import MagicMock
import sys
import os

# Add the project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from crowd_pilot.serialization_utils import ConversationState

class MockTokenizer:
    def __init__(self):
        pass
    
    def encode(self, text):
        # Mock tokenization: 1 char = 1 token for simplicity in tests
        return list(text)
        
    def decode(self, tokens):
        return "".join(tokens)

class TestConversationState(unittest.TestCase):
    def setUp(self):
        self.tokenizer = MockTokenizer()
        self.conversations = []
        self.max_tokens_per_conversation = 100
        self.max_tokens_per_message = 50
        self.min_conversation_messages = 2
        
        self.state = ConversationState(
            conversations=self.conversations,
            max_tokens_per_conversation=self.max_tokens_per_conversation,
            max_tokens_per_message=self.max_tokens_per_message,
            min_conversation_messages=self.min_conversation_messages,
            tokenizer=self.tokenizer
        )

    def test_renaming_and_logic_dropped(self):
        """Test 1: Create a conversation with 6 Assistant messages and 0 User messages. Verify it is DROPPED."""
        # 6 Assistant messages
        for i in range(6):
            self.state.append_message({"from": "Assistant", "value": f"cmd{i}"})
            
        # Finalize
        self.state.finalize_conversation()
        
        # Should be dropped because no User role
        self.assertEqual(len(self.conversations), 0)

    def test_logic_kept(self):
        """Test 2: Create a conversation with 5 Assistant messages and 1 User message. Verify it is KEPT."""
        # 5 Assistant messages
        for i in range(5):
            self.state.append_message({"from": "Assistant", "value": f"cmd{i}"})
        # 1 User message
        self.state.append_message({"from": "User", "value": "output"})
        
        self.state.finalize_conversation()
        
        # Should be kept
        self.assertEqual(len(self.conversations), 1)
        self.assertEqual(len(self.conversations[0]), 6)

    def test_token_counting(self):
        """Test 3: Verify limit is respected based on TOKENS."""
        self.state = ConversationState(
            conversations=self.conversations,
            max_tokens_per_conversation=100,
            max_tokens_per_message=100,
            min_conversation_messages=1,
            tokenizer=self.tokenizer
        )
        
        # Msg 1: User (40 tokens)
        self.state.append_message({"from": "User", "value": "a" * 40})
        # Msg 2: Assistant (40 tokens) -> Total 80. OK.
        self.state.append_message({"from": "Assistant", "value": "b" * 40})
        
        # Msg 3: User (30 tokens) -> Total 110. Should split.
        # The split happens BEFORE adding Msg 3.
        # So Conversation 1 = Msg 1 + Msg 2 (80 tokens).
        # Conversation 2 (current) = Msg 3.
        self.state.append_message({"from": "User", "value": "c" * 30})
        
        self.assertEqual(len(self.conversations), 1)
        self.assertEqual(len(self.conversations[0]), 2) # Msg 1 + Msg 2
        self.assertEqual(self.state.current_tokens, 30) # Msg 3
        self.assertEqual(len(self.state.current_conversation), 1)

    def test_both_roles_required(self):
        """Test 4: Create a conversation with User messages but no Assistant messages. Verify it is DROPPED."""
        self.state.append_message({"from": "User", "value": "output1"})
        self.state.append_message({"from": "User", "value": "output2"})
        
        self.state.finalize_conversation()
        
        self.assertEqual(len(self.conversations), 0)

    def test_min_messages_respected(self):
        """Test 5: Verify min_conversation_messages is respected."""
        self.state.min_conversation_messages = 5
        
        # 2 messages (User + Assistant) -> Valid roles, but too short
        self.state.append_message({"from": "User", "value": "out"})
        self.state.append_message({"from": "Assistant", "value": "cmd"})
        
        self.state.finalize_conversation()
        
        self.assertEqual(len(self.conversations), 0)

if __name__ == "__main__":
    unittest.main()
