import asyncio
import aiohttp
import backoff
from typing import Optional, Dict, Any
import openai
import anthropic
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from .types import LLMRequest, LLMResponse
from .config import settings
from .utils.logging import logger

class LLMClientError(Exception):
    """Custom exception for LLM client errors."""
    pass

class LLMClient:
    """Unified client for multiple LLM providers with retry logic and failover."""
    
    def __init__(self):
        self.openai_client = None
        self.anthropic_client = None
        
        if settings.openai_api_key:
            self.openai_client = AsyncOpenAI(api_key=settings.openai_api_key)
        
        if settings.anthropic_api_key:
            self.anthropic_client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    
    @backoff.on_exception(
        backoff.expo,
        (aiohttp.ClientError, openai.APIError, anthropic.APIError),
        max_tries=settings.retry_attempts,
        factor=settings.backoff_factor
    )
    async def generate_response(self, request: LLMRequest) -> LLMResponse:
        """Generate response from LLM with retry logic and provider failover."""
        provider = request.provider or settings.primary_llm_provider
        
        try:
            if provider == "openai":
                return await self._call_openai(request)
            elif provider == "anthropic":
                return await self._call_anthropic(request)
            else:
                raise LLMClientError(f"Unknown provider: {provider}")
                
        except Exception as e:
            logger.error("Primary provider failed", provider=provider, error=str(e))
            
            # Try fallback provider
            fallback_provider = settings.fallback_llm_provider
            if fallback_provider != provider:
                logger.info("Attempting fallback provider", fallback_provider=fallback_provider)
                request.provider = fallback_provider
                request.model = settings.fallback_model
                
                if fallback_provider == "openai":
                    return await self._call_openai(request)
                elif fallback_provider == "anthropic":
                    return await self._call_anthropic(request)
            
            raise LLMClientError(f"All providers failed. Last error: {str(e)}")
    
    async def _call_openai(self, request: LLMRequest) -> LLMResponse:
        """Call OpenAI API."""
        if not self.openai_client:
            raise LLMClientError("OpenAI client not initialized")
        
        try:
            response = await self.openai_client.chat.completions.create(
                model=request.model,
                messages=request.messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens
            )
            
            return LLMResponse(
                content=response.choices[0].message.content,
                usage_tokens=response.usage.total_tokens,
                model_used=request.model,
                provider_used="openai"
            )
        except Exception as e:
            logger.error("OpenAI API call failed", error=str(e))
            raise
    
    async def _call_anthropic(self, request: LLMRequest) -> LLMResponse:
        """Call Anthropic API."""
        if not self.anthropic_client:
            raise LLMClientError("Anthropic client not initialized")
        
        try:
            # Convert messages format for Anthropic
            system_message = ""
            messages = []
            
            for msg in request.messages:
                if msg["role"] == "system":
                    system_message = msg["content"]
                else:
                    messages.append(msg)
            
            response = await self.anthropic_client.messages.create(
                model=request.model,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                system=system_message,
                messages=messages
            )
            
            return LLMResponse(
                content=response.content[0].text,
                usage_tokens=response.usage.input_tokens + response.usage.output_tokens,
                model_used=request.model,
                provider_used="anthropic"
            )
        except Exception as e:
            logger.error("Anthropic API call failed", error=str(e))
            raise