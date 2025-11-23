import base64
import io
import logging
import os
from typing import Dict, Tuple

from openai import OpenAI


logger = logging.getLogger(__name__)


def _get_openai_client(api_key: str | None = None) -> OpenAI:
    """
    Create an OpenAI client using the provided API key or the OPENAI_API_KEY environment variable.
    
    Parameters
    ----------
    api_key:
        Optional OpenAI API key. If provided, this will be used.
        If None, falls back to OPENAI_API_KEY environment variable.
    """
    if api_key:
        return OpenAI(api_key=api_key)
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OpenAI API key is required. "
            "Please provide it in the app settings or set OPENAI_API_KEY environment variable."
        )

    return OpenAI(api_key=api_key)


def transcribe_audio(
    audio_file: io.BufferedIOBase,
    model: str = "whisper-1",
    api_key: str | None = None,
) -> str:
    """
    Transcribe an audio file to text using OpenAI Whisper.

    Parameters
    ----------
    audio_file:
        A binary file-like object positioned at the beginning, with a valid
        `.name` attribute that includes an audio file extension.
    model:
        The OpenAI audio transcription model to use.
    api_key:
        Optional OpenAI API key. If not provided, uses environment variable.
    """
    client = _get_openai_client(api_key=api_key)
    logger.info("Starting transcription with model=%s", model)

    # OpenAI client expects a file-like object; make sure we're at the start.
    audio_file.seek(0)

    transcript = client.audio.transcriptions.create(
        model=model,
        file=audio_file,
        response_format="json",
        temperature=0,
    )

    text = transcript.text.strip()
    logger.info("Transcription finished. Length=%d characters", len(text))
    
    if not text:
        raise ValueError(
            "Transcription returned empty text. The audio file may be too quiet, "
            "corrupted, or contain no speech. Please try with a different audio file."
        )
    
    return text


def build_image_prompt(
    transcript: str,
    model: str = "gpt-4o-mini",
    api_key: str | None = None,
) -> str:
    """
    Use an LLM to convert a raw user transcript into a detailed image prompt.

    The returned text is intended to be passed directly to the image
    generation model.
    
    Parameters
    ----------
    transcript:
        The transcribed text from the audio file.
    model:
        The OpenAI chat model to use for prompt generation.
    api_key:
        Optional OpenAI API key. If not provided, uses environment variable.
    """
    # Validate transcript is not empty
    if not transcript or not transcript.strip():
        raise ValueError(
            "Transcript is empty. Cannot generate image prompt from empty transcript."
        )

    client = _get_openai_client(api_key=api_key)
    logger.info("Starting prompt generation with model=%s", model)
    logger.info("Transcript received: %s", transcript[:100] + "..." if len(transcript) > 100 else transcript)

    # Enhanced system message with more explicit instructions
    system_message = (
        "You are a creative assistant that transforms voice requests into detailed, "
        "concrete image generation prompts. Your task is CRITICAL: you MUST always return "
        "a non-empty, descriptive image prompt.\n\n"
        "Guidelines:\n"
        "- Focus purely on describing a visual scene, not dialogue or conversation\n"
        "- Include specific details: subject, style, lighting, colors, composition, "
        "camera angle, and atmosphere\n"
        "- Keep it concise but vivid (1-2 sentences)\n"
        "- If the transcript is unclear, interpret it creatively and describe a reasonable visual scene\n"
        "- NEVER return empty text, whitespace only, or just punctuation\n"
        "- Your response must be a complete, usable image generation prompt\n\n"
        "CRITICAL: You must ALWAYS return a non-empty prompt. Even if the transcript is unclear, "
        "create a descriptive visual prompt based on your best interpretation."
    )

    user_message = (
        "User voice transcript:\n"
        f"\"{transcript}\"\n\n"
        "Transform this into a single, self-contained image generation prompt that "
        "captures the user's intent as a visual scene. Return ONLY the prompt text, "
        "nothing else. Make it detailed and visually descriptive."
    )

    # Try with the primary prompt first
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            temperature=0.8,  # Slightly lower for more consistency
            max_tokens=300,
        )

        prompt = completion.choices[0].message.content
        if prompt is None:
            logger.warning("LLM returned None, attempting retry with fallback approach")
            raise ValueError("LLM returned None")
        
        prompt = prompt.strip()
        
        # If empty, try a more direct approach
        if not prompt:
            logger.warning("LLM returned empty prompt, attempting retry with simpler approach")
            raise ValueError("Empty prompt returned")
        
        logger.info("Prompt generation finished. Length=%d characters", len(prompt))
        return prompt
    
    except (ValueError, Exception) as e:
        # Retry with a simpler, more direct approach
        logger.info("Retrying prompt generation with fallback approach")
        
        fallback_system = (
            "You are an image prompt generator. Transform the given text into a visual description. "
            "Always return a descriptive image prompt, even if the input is unclear. "
            "Make it creative and detailed."
        )
        
        fallback_user = (
            f"Create an image generation prompt from this text: \"{transcript}\"\n\n"
            "Return a detailed visual description suitable for image generation. "
            "Include subject, style, and visual details."
        )
        
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": fallback_system},
                    {"role": "user", "content": fallback_user},
                ],
                temperature=0.7,
                max_tokens=250,
            )
            
            prompt = completion.choices[0].message.content
            if prompt:
                prompt = prompt.strip()
                if prompt:
                    logger.info("Fallback prompt generation succeeded. Length=%d characters", len(prompt))
                    return prompt
        except Exception as retry_error:
            logger.warning("Fallback attempt also failed: %s", retry_error)
        
        # Final fallback: create a basic prompt from the transcript
        logger.info("Using final fallback: creating prompt directly from transcript")
        fallback_prompt = (
            f"A detailed, artistic visualization of: {transcript.strip()}. "
            "High quality, vivid colors, professional composition, cinematic lighting."
        )
        logger.info("Using fallback prompt. Length=%d characters", len(fallback_prompt))
        return fallback_prompt


def generate_image(
    prompt: str,
    model: str = "gpt-image-1",
    size: str = "1024x1024",
    api_key: str | None = None,
) -> Tuple[bytes, Dict[str, str]]:
    """
    Generate an image from a text prompt using OpenAI's image model.

    Parameters
    ----------
    prompt:
        The text prompt describing the image to generate.
    model:
        The OpenAI image generation model to use.
    size:
        The size of the generated image (e.g., "1024x1024").
    api_key:
        Optional OpenAI API key. If not provided, uses environment variable.

    Returns
    -------
    image_bytes:
        The raw bytes of the generated image (PNG by default).
    metadata:
        A small dict with additional information (e.g., model, size).
    """
    # Validate prompt is not empty
    if not prompt or not prompt.strip():
        raise ValueError(
            "Image prompt is empty. Cannot generate image from empty prompt. "
            "Please ensure the transcript was properly transcribed and the LLM generated a valid prompt."
        )

    client = _get_openai_client(api_key=api_key)
    logger.info(
        "Starting image generation with model=%s size=%s", model, size
    )

    result = client.images.generate(
        model=model,
        prompt=prompt,
        size=size,
        n=1,
    )

    image_b64 = result.data[0].b64_json
    image_bytes = base64.b64decode(image_b64)

    logger.info("Image generation finished. Bytes=%d", len(image_bytes))
    return image_bytes, {"model": model, "size": size}


