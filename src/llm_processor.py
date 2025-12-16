import os
import torch
import warnings
from typing import Dict, Any, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from dotenv import load_dotenv
from transformers import BitsAndBytesConfig
import re

warnings.filterwarnings('ignore', category=DeprecationWarning, module='pkg_resources')
load_dotenv()


class LLMProcessor:
    """
    Processes queries and generates natural language responses using Phi-3-mini LLM.
    Implements singleton pattern for efficient model reuse.
    """
    
    _instance = None
    _client = None
    _device = None
    _tokenizer = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(LLMProcessor, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, 
                 model_name: str = "microsoft/Phi-3-mini-4k-instruct",
                 temperature: float = 0.6,
                 max_tokens: int = 150):
        if self._initialized:
            return
        
        if LLMProcessor._client is None:
            print(f"Initializing Phi-3-mini model: {model_name}")
            print("⏳ Loading model and tokenizer...")
            
            try:
                LLMProcessor._tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    trust_remote_code=True
                )
                
                if torch.cuda.is_available():
                    LLMProcessor._device = "cuda"
                    print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
                else:
                    LLMProcessor._device = "cpu"
                    print(f"✓ Using CPU")
                
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )

                if LLMProcessor._device == "cuda":
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        dtype=torch.float16, 
                        device_map="auto",
                        quantization_config=quantization_config,
                        trust_remote_code=True
                    )
                    LLMProcessor._client = pipeline(
                        "text-generation",
                        model=model,
                        tokenizer=LLMProcessor._tokenizer
                    )
                else:
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        dtype=torch.float32,  
                        trust_remote_code=True,
                        attn_implementation="eager",
                        quantization_config=quantization_config,
                        _attn_implementation_internal="eager"
                    )
                    model = model.to(LLMProcessor._device)
                    LLMProcessor._client = pipeline(
                        "text-generation",
                        model=model,
                        tokenizer=LLMProcessor._tokenizer,
                        device=-1
                    )
                
                print("✓ Phi-3-mini model initialized successfully!")
                
            except Exception as e:
                raise RuntimeError(f"Failed to initialize Phi-3-mini model: {e}")
        else:
            print("✓ Reusing cached Phi-3-mini model")
        
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._initialized = True
    
    def generate_response(self, 
                         query: str, 
                         data_context: str,
                         intent: Optional[str] = None,
                         conversation_history: Optional[str] = None) -> str:
        """Generate natural language response based on query and data."""
        try:
            return self._generate_with_phi3(query, data_context, intent, conversation_history)
        except Exception as e:
            print(f"❌ Error generating LLM response: {e}")
            return f"I encountered an error: {str(e)}"
    
    def _generate_with_phi3(self, 
                           query: str, 
                           data_context: str,
                           intent: Optional[str] = None,
                           conversation_history: Optional[str] = None) -> str:
        """Generate response using Phi-3-mini model."""
        
        # Build optimized prompt
        system_prompt = self._get_system_prompt(intent)
        user_message = self._build_user_message(query, data_context, conversation_history)
        
        full_prompt = f"""<|system|>
{system_prompt}<|end|>
<|user|>
{user_message}<|end|>
<|assistant|>
"""
        
        # Generate with optimized settings
        response = LLMProcessor._client(
            full_prompt,
            max_new_tokens=self.max_tokens,
            temperature=self.temperature,
            do_sample=True,
            top_p=0.9,
            return_full_text=False,
            use_cache=False,
            pad_token_id=LLMProcessor._tokenizer.eos_token_id
        )
        
        return self._clean_response(response[0]['generated_text'])
    
    def _get_system_prompt(self, intent: Optional[str] = None) -> str:
        """Build concise system prompt."""
        base = """You are a health data assistant. Report numbers from the provided data.

RULES:
1. ALWAYS include the exact numeric values from the data
2. NEVER give generic responses without numbers
3. Be brief (1-2 sentences) but include the specific value
4. For "current" queries, use the latest data shown
5. for 1 day don't tell average value

Example: Data shows "8,832 steps" → Answer: "You have 8,832 steps today." """
        
        intent_suffix = {
            "query_steps": "Focus on step count.",
            "query_calories": "Focus on calories.",
            "query_sleep": "Focus on sleep hours.",
            "query_heart_rate": "Focus on heart rate (BPM).",
            "comparison": "State both values and difference.",
            "trend_analysis": "Mention actual values in pattern."
        }
        
        return base + intent_suffix.get(intent, "")
    
    def _build_user_message(self, query: str, data_context: str, history: Optional[str] = None) -> str:
        """Build optimized user message."""
        parts = []
        
        if history:
            parts.append(f"Previous:\n{history}\n")
        
        parts.extend([
            "DATA:",
            data_context,
            "",
            f"Question: {query}",
            "Answer using the specific numbers from DATA above in 1-2 sentences."
        ])
        
        return "\n".join(parts)
    
    def _clean_response(self, response: str) -> str:
        """Clean and format response efficiently."""
        response = response.strip()
        
        # Remove formatting
        for token in ('**', '*', '- ', '• '):
            response = response.replace(token, '')
        
        # Remove verbose prefixes
        prefixes = ["Based on the data provided, ", "According to the data, ", 
                   "Looking at the data, ", "From the data, "]
        for prefix in prefixes:
            if response.startswith(prefix):
                response = response[len(prefix):]
                break
        
        # Limit to 2 sentences
        parts = [p.strip() for p in re.split(r'(?<=[\.\?\!])\s+', response) if p.strip()]
        if len(parts) > 2:
            response = ' '.join(parts[:2])
        
        # Ensure ending punctuation
        if response and not response.endswith(('.', '!', '?')):
            response += '.'
        
        return response
    
    @classmethod
    def get_client_info(cls) -> Dict[str, Any]:
        """Get model information."""
        return {
            'client_initialized': cls._client is not None,
            'model': 'microsoft/Phi-3-mini-4k-instruct',
            'device': cls._device or 'Not initialized',
            'cuda_available': torch.cuda.is_available()
        }
    
    @classmethod
    def reset_client(cls):
        """Reset the model (for testing/debugging)."""
        if cls._client is not None:
            print("Resetting Phi-3 model...")
            del cls._client
            cls._client = None
            cls._device = None
            cls._tokenizer = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("✓ Model reset!")


# ================== TESTING ==================

if __name__ == "__main__":
    print("="*60)
    print("Testing Optimized LLM Processor")
    print("="*60)
    
    processor = LLMProcessor()
    
    # Test 1: Current step count (your problematic query)
    test_data1 = """📊 STEPS
📅 Date: 2025-11-07 @ 14:45
Total: 8,832 steps
Distance: 3.98 km"""
    
    print("\n" + "="*60)
    print("Test 1: Current Step Count")
    print("="*60)
    print(f"Query: What is my current step count?")
    print(f"Data:\n{test_data1}")
    
    response1 = processor.generate_response(
        "What is my current step count?", 
        test_data1, 
        intent="query_steps"
    )
    print(f"\n🤖 Response: {response1}")
    print(f"✓ Contains number: {'8,832' in response1 or '8832' in response1}")
    
    # Test 2: Heart rate
    test_data2 = """📊 HEART_RATE
📅 Date: 2025-10-05
Average: 74.0 bpm
Resting: 64.8 bpm"""
    
    print("\n" + "="*60)
    print("Test 2: Heart Rate")
    print("="*60)
    print(f"Query: How was my heart rate on Sunday?")
    print(f"Data:\n{test_data2}")
    
    response2 = processor.generate_response(
        "How was my heart rate on Sunday?", 
        test_data2, 
        intent="query_heart_rate"
    )
    print(f"\n🤖 Response: {response2}")
    print(f"✓ Contains number: {'74' in response2 or '64.8' in response2}")
    
    # Test 3: With history
    history = "User: What is my current step count?\nAssistant: You have 8,832 steps today."
    test_data3 = """📊 STEPS
📅 Date: 2025-10-07
Total: 12,450 steps"""
    
    print("\n" + "="*60)
    print("Test 3: With Conversation History")
    print("="*60)
    print(f"Query: What about Tuesday?")
    print(f"Data:\n{test_data3}")
    
    response3 = processor.generate_response(
        "What about Tuesday?",
        test_data3,
        intent="query_steps",
        conversation_history=history
    )
    print(f"\n🤖 Response: {response3}")
    print(f"✓ Contains number: {'12,450' in response3 or '12450' in response3}")
    
    print("\n" + "="*60)
    print("✓ Testing Complete!")
    print("="*60)