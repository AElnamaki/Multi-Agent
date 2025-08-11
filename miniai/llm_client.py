"""
Async LLM client with retry logic and structured output parsing
"""

import json
import re
import httpx
import asyncio
from typing import Dict, Any, Optional, List
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from .config import Config
from .observability import get_logger
import time
import uuid

logger = get_logger(__name__)


class LLMError(Exception):
    """Base exception for LLM errors"""
    pass


class LLMParseError(LLMError):
    """Error parsing LLM response"""
    pass


class LLMTimeoutError(LLMError):
    """LLM request timeout"""
    pass


class AsyncLLMClient:
    """Async LLM client with retry and structured output"""
    
    def __init__(self, config: Config):
        self.config = config
        self.client = httpx.AsyncClient(timeout=config.llm.timeout)
        self._request_count = 0
    
    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=0.5, max=60),
        retry=retry_if_exception_type((httpx.RequestError, LLMTimeoutError))
    )
    async def _make_request(self, messages: List[Dict[str, str]], 
                           max_tokens: Optional[int] = None) -> Dict[str, Any]:
        """Make HTTP request to LLM provider"""
        self._request_count += 1
        request_id = str(uuid.uuid4())[:8]
        
        start_time = time.time()
        
        try:
            if self.config.llm.provider == "openai":
                response = await self._openai_request(messages, max_tokens, request_id)
            elif self.config.llm.provider == "anthropic":
                response = await self._anthropic_request(messages, max_tokens, request_id)
            else:
                raise LLMError(f"Unsupported provider: {self.config.llm.provider}")
            
            duration = time.time() - start_time
            logger.info(
                "LLM request completed",
                extra={
                    "request_id": request_id,
                    "provider": self.config.llm.provider,
                    "duration": duration,
                    "tokens": response.get("usage", {}).get("total_tokens", 0)
                }
            )
            
            return response
            
        except httpx.TimeoutException as e:
            logger.error(f"LLM request timeout: {e}", extra={"request_id": request_id})
            raise LLMTimeoutError(str(e))
        except Exception as e:
            logger.error(f"LLM request failed: {e}", extra={"request_id": request_id})
            raise
    
    async def _openai_request(self, messages: List[Dict[str, str]], 
                             max_tokens: Optional[int], request_id: str) -> Dict[str, Any]:
        """Make request to OpenAI API"""
        headers = {
            "Authorization": f"Bearer {self.config.llm.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.config.llm.model,
            "messages": messages,
            "temperature": self.config.llm.temperature,
            "max_tokens": max_tokens or self.config.llm.max_tokens
        }
        
        base_url = self.config.llm.base_url or "https://api.openai.com/v1"
        
        response = await self.client.post(
            f"{base_url}/chat/completions",
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        return response.json()
    
    async def _anthropic_request(self, messages: List[Dict[str, str]], 
                                max_tokens: Optional[int], request_id: str) -> Dict[str, Any]:
        """Make request to Anthropic API"""
        headers = {
            "x-api-key": self.config.llm.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        
        # Convert messages format for Anthropic
        system_message = ""
        user_messages = []
        
        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                user_messages.append(msg)
        
        payload = {
            "model": self.config.llm.model,
            "messages": user_messages,
            "max_tokens": max_tokens or self.config.llm.max_tokens,
            "temperature": self.config.llm.temperature
        }
        
        if system_message:
            payload["system"] = system_message
        
        base_url = self.config.llm.base_url or "https://api.anthropic.com"
        
        response = await self.client.post(
            f"{base_url}/v1/messages",
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        return response.json()
    
    def _extract_json_from_text(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract JSON from text using multiple strategies"""
        # Strategy 1: Try direct JSON parse
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass
        
        # Strategy 2: Look for JSON in code blocks
        json_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
        matches = re.findall(json_pattern, text, re.DOTALL | re.IGNORECASE)
        if matches:
            try:
                return json.loads(matches[-1])
            except json.JSONDecodeError:
                pass
        
        # Strategy 3: Find last complete JSON object
        brace_count = 0
        json_start = -1
        json_end = -1
        
        for i, char in enumerate(text):
            if char == '{':
                if brace_count == 0:
                    json_start = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and json_start != -1:
                    json_end = i + 1
        
        if json_start != -1 and json_end != -1:
            try:
                return json.loads(text[json_start:json_end])
            except json.JSONDecodeError:
                pass
        
        return None
    
    async def generate_structured(self, messages: List[Dict[str, str]], 
                                 schema_description: str,
                                 max_tokens: Optional[int] = None) -> Dict[str, Any]:
        """Generate structured output with JSON schema validation"""
        
        # Add JSON instruction to the last message
        if messages and messages[-1]["role"] == "user":
            messages[-1]["content"] += f"\n\nIMPORTANT: Return ONLY a valid JSON object that matches this schema: {schema_description}"
        
        response = await self._make_request(messages, max_tokens)
        
        # Extract content based on provider
        if self.config.llm.provider == "openai":
            content = response["choices"][0]["message"]["content"]
        elif self.config.llm.provider == "anthropic":
            content = response["content"][0]["text"]
        else:
            raise LLMError(f"Unsupported provider: {self.config.llm.provider}")
        
        # Parse JSON from response
        parsed_json = self._extract_json_from_text(content)
        
        if parsed_json is None:
            # Try correction prompt
            logger.warning("Failed to extract JSON, attempting correction")
            correction_messages = [
                {
                    "role": "user", 
                    "content": f"The previous response was not valid JSON. Please provide ONLY a valid JSON object matching this schema: {schema_description}\n\nPrevious response: {content}"
                }
            ]
            
            correction_response = await self._make_request(correction_messages, max_tokens)
            
            if self.config.llm.provider == "openai":
                correction_content = correction_response["choices"][0]["message"]["content"]
            else:
                correction_content = correction_response["content"][0]["text"]
            
            parsed_json = self._extract_json_from_text(correction_content)
            
            if parsed_json is None:
                raise LLMParseError(f"Failed to parse JSON from LLM response: {content}")
        
        return parsed_json
