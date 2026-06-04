#!/usr/bin/env python3
"""
NIM Client — OpenAI-compatible client for NVIDIA NIM API.
Handles text generation, vision analysis, image generation, and
LinkedIn post creation with user profile personalization.

Models:
- llama-3.3-70b-instruct (text generation)
- llama-3.2-90b-vision-instruct (image analysis)
- stable-diffusion-xl (image generation)
"""

import base64
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from openai import OpenAI
except ImportError:
    print("Missing dependency: openai")
    print("Run: pip install openai")
    sys.exit(1)

BASE_DIR = Path(__file__).parent


def encode_image(image_path: str) -> str:
    """Encode an image file to base64 string."""
    with open(image_path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def get_image_extension(image_path: str) -> str:
    ext = Path(image_path).suffix.lower()
    mime_map = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }
    return mime_map.get(ext, "image/jpeg")


def load_history_topics() -> List[str]:
    """Load recent post topics from posts.json to avoid repetition."""
    posts_path = BASE_DIR / "posts.json"
    if not posts_path.exists():
        return []
    try:
        with open(posts_path, "r", encoding="utf-8") as f:
            posts = json.load(f)
        return [p.get("title", "") for p in posts[-10:]]
    except (json.JSONDecodeError, IOError):
        return []


class NIMClient:
    """Client for interacting with NVIDIA NIM API."""

    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = BASE_DIR / "nim_config.json"

        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        # Allow env var override (useful for one-off runs without editing config)
        api_key = os.environ.get("NIM_API_KEY") or self.config.get("api_key", "")
        if not api_key:
            raise ValueError(
                "No API key found. Set it in nim_config.json or via NIM_API_KEY env var.\n"
                "Get your key at: https://build.nvidia.com"
            )

        base_url = self.config.get("base_url", "https://integrate.api.nvidia.com/v1")

        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.text_model = self.config.get("text_model", "meta/llama-3.3-70b-instruct")
        self.vision_model = self.config.get(
            "vision_model", "meta/llama-3.2-90b-vision-instruct"
        )
        self.image_model = self.config.get("image_model", "stable-diffusion-xl")
        self.profile: Dict[str, Any] = self.config.get("profile_context", {})

    # ------------------------------------------------------------------
    # Profile context management
    # ------------------------------------------------------------------

    def set_user_context(self, profile_dict: Dict[str, Any]) -> None:
        """Set or update the user profile used for post generation.

        Call this before generating posts to personalise content.
        Mutates the in-memory profile and persists to nim_config.json.

        Args:
            profile_dict: Dict with keys like 'name', 'role', 'field',
                          'interests' (list), 'tone', 'goals', etc.
        """
        self.profile.update(profile_dict)
        self.config["profile_context"] = self.profile
        with open(BASE_DIR / "nim_config.json", "w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)

    def set_profile_context(self, **kwargs) -> None:
        """Update profile context for more tailored generations (backward-compat)."""
        self.profile.update(kwargs)
        self.config["profile_context"] = self.profile
        with open(BASE_DIR / "nim_config.json", "w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)

    def get_profile_context(self) -> Dict[str, Any]:
        """Get current profile context."""
        return self.profile

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _chat(self, model: str, messages: list, **kwargs) -> str:
        """Internal chat completion call."""
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=kwargs.get("max_tokens", 1024),
                temperature=kwargs.get("temperature", 0.7),
                top_p=kwargs.get("top_p", 0.95),
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            error_msg = str(e)
            if "401" in error_msg or "Unauthorized" in error_msg:
                return (
                    "ERROR: API key invalid or expired. Get a key at https://build.nvidia.com\n"
                    f"Details: {error_msg}"
                )
            if "429" in error_msg:
                return (
                    "ERROR: Rate limited. Wait a moment and try again.\n"
                    f"Details: {error_msg}"
                )
            return f"ERROR calling NIM: {error_msg}"

    def _build_profile_block(self) -> str:
        """Build a descriptive paragraph from the stored profile dict."""
        name = self.profile.get("name", "a professional")
        role = self.profile.get("role", "professional")
        field = self.profile.get("field", "technology")
        interests = ", ".join(self.profile.get("interests", [])) or "technology and growth"
        tone = self.profile.get("tone", "professional but approachable")
        goals_list = self.profile.get("goals", [])
        goals = ", ".join(goals_list) if goals_list else "sharing insights and growing their network"
        return (
            f"You are {name}, a {role} working in {field}.\n"
            f"Your interests: {interests}.\n"
            f"Your LinkedIn goals: {goals}.\n"
            f"Write in a {tone} tone.\n"
        )

    def _build_linkedin_rules(self) -> str:
        """Shared LinkedIn best-practice rules and human-like writing directives."""
        return (
            "LinkedIn Formatting & Structure Rules:\n"
            "- Hook in the very first line: start in media res, share a surprising statistic, a bold assertion, or a personal realization. Never start with a question (e.g. 'Have you ever...') or a generic introduction.\n"
            "- Use varying sentence and paragraph lengths: mix short, punchy single-clause sentences with slightly longer, natural sentences. Keep paragraphs between 1-3 sentences.\n"
            "- Tone & Voice: write in first-person ('I', 'my', 'we'). Sound like a human sharing a real conversation with a peer over coffee. Be conversational, direct, and slightly opinionated.\n"
            "- Emojis: use at most 1-2 emojis per post, or none. Never start lines with emojis as bullet points. Never use emojis as punctuation replacements.\n"
            "- No Unicode Bold Formatting: do not use unicode bold font for words (like 𝗯𝗼𝗹𝗱) as it is hard to read and looks spammy.\n"
            "- Call to Action (CTA): end with a natural, open-ended question that invites others to share their experience. Avoid generic prompts like 'What do you think? Let me know below!'\n"
            "- Hashtags: include exactly 3-5 relevant, lowercase hashtags separated by spaces on the very last line, after a blank line. Do not embed hashtags in the body of the post.\n"
            "- Word Count: keep the post body concise, between 120 and 250 words.\n"
            "\n"
            "CRITICAL: Avoid AI clichés, template phrases, and generic patterns:\n"
            "- Never use transition words like 'Moreover', 'Furthermore', 'In conclusion', 'In today's fast-paced world', 'In the evolving landscape', 'Testament to', 'Embark on a journey', 'Pivotal', 'Revolutionize', 'Tapestry', 'Beacon', 'Unlocking'.\n"
            "- Avoid hollow words: 'leverage' (use 'use' or 'apply'), 'utilize', 'robust' (use 'reliable' or 'strong'), 'streamline' (use 'simplify'), 'foster', 'seamless', 'delve', 'demystify'.\n"
            "- Do not use synonym cycling, rule-of-three list patterns, or generic summaries at the end.\n"
            "\n"
            "Detailed AI Image Prompts Guidelines:\n"
            "- The 'image_prompts' array MUST contain exactly one extremely detailed and vivid image prompt designed for tools like Stable Diffusion, DALL-E, or Midjourney.\n"
            "- The prompt must describe a modern, premium, and professional scene: specify the subject, composition (e.g. close-up shot, rule of thirds, clean overhead desk layout), background details, lighting style (e.g. soft volumetric lighting, warm natural light from a window, subtle lens flare), mood, art style (e.g. professional editorial photography, clean minimal vector illustration, cinematic bokeh style), color palette (e.g. soft earth tones, cool tech blues and slate gray), and technical camera parameters (e.g. shot on 35mm lens, f/1.8 aperture, sharp focus on subject, blurred background).\n"
            "- Avoid simple or generic descriptions like 'a picture of code' or 'a team meeting'."
        )

    # ------------------------------------------------------------------
    # Post generation
    # ------------------------------------------------------------------

    def generate_post(
        self,
        topic: Optional[str] = None,
        style: Optional[str] = None,
        user_profile: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Generate a LinkedIn post personalized to the user's profile.

        Args:
            topic: Specific topic to write about (optional).
            style: Writing style hint — 'story', 'tip', 'opinion', 'question'.
            user_profile: Override profile for this generation.  Keys: name,
                          role, field, interests (list), tone, goals (list).

        Returns dict with 'title', 'content', 'hashtags', 'image_prompts'.
        """
        # Allow per-call profile override without mutating state
        original_profile = self.profile
        if user_profile:
            self.profile = {**self.profile, **user_profile}

        try:
            name = self.profile.get("name", "a professional")
            role = self.profile.get("role", "professional")
            field = self.profile.get("field", "technology")
            interests = ", ".join(self.profile.get("interests", [])) or "technology and growth"
            tone = self.profile.get("tone", "professional but approachable")
            goals_list = self.profile.get("goals", ["sharing insights and growing their network"])
            goals = ", ".join(goals_list)

            past_titles = load_history_topics()
            avoid_block = ""
            if past_titles:
                avoid_block = (
                    f"\nDo NOT reuse topics from these previous posts: {', '.join(past_titles)}.\n"
                    "Find a fresh angle or a new topic entirely.\n"
                )

            style_hints = {
                "story": (
                    "Tell a short personal story with a clear takeaway. "
                    "Start with a hook, include a moment of realization."
                ),
                "tip": (
                    "Share a practical, actionable tip. "
                    "Be specific and include steps or a framework."
                ),
                "opinion": (
                    "Share a strong but respectful opinion on something "
                    "in your field. Invite discussion at the end."
                ),
                "question": (
                    "Pose an interesting question to spark discussion. "
                    "Share your own perspective first, then ask."
                ),
            }

            style_block = ""
            if style and style in style_hints:
                style_block = style_hints[style] + "\n"

            topic_instruction = (
                f"Topic to write about: {topic}" if topic
                else "Pick an engaging, relevant topic that would resonate with "
                "professionals in your field. Prefer topics tied to your stated interests "
                f"({interests}) and your goals ({goals}). Something timely, personal, "
                "or insightful."
            )

            context_block = f"""You are {name}, a {role} working in {field}.
Your interests: {interests}.
Your LinkedIn goals: {goals}.
Write in a {tone} tone.

{topic_instruction}
{style_block}
{avoid_block}
{self._build_linkedin_rules()}

Respond in this exact JSON format:
{{
  "title": "Short title for your reference (max 60 chars)",
  "content": "The full post text with line breaks",
  "hashtags": ["tag1", "tag2", "tag3"],
  "image_prompts": ["detailed prompt for an AI image that would complement this post"]
}}""".strip()

            result = self._chat(
                model=self.text_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an elite LinkedIn content strategist and professional copywriter. "
                            "You write highly engaging, human-sounding posts that completely avoid AI clichés, "
                            "monotonous structures, and typical 'AI-isms'. Your posts feel authentic, personal, "
                            "and conversational."
                        )
                    },
                    {"role": "user", "content": context_block},
                ],
                temperature=0.8,
                max_tokens=800,
            )
            return self._parse_post_response(result)
        finally:
            if user_profile:
                self.profile = original_profile

    def generate_weekly_batch(
        self,
        topics: List[str],
        user_profile: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Generate a batch of posts for weekly scheduling.

        Each post is personalised with the user's profile.  Topics should be
        varied so the week feels diverse.  The method deduplicates against
        the existing post history automatically.

        Args:
            topics: List of topic strings (one per desired post).
            user_profile: Optional dict to override stored profile for this batch.

        Returns list of post dicts (same shape as generate_post).
        """
        posts: List[Dict[str, Any]] = []
        for topic in topics:
            post = self.generate_post(topic=topic, user_profile=user_profile)
            posts.append(post)
        return posts

    # ------------------------------------------------------------------
    # Post analysis & improvement
    # ------------------------------------------------------------------

    def generate_variations(self, base_content: str, count: int = 2) -> list[dict]:
        """Generate alternative versions of an existing post draft."""
        prompt = f"""\
Given this LinkedIn post draft, generate {count} alternative versions that keep the same \
core message but change the angle, hook, or structure for different engagement styles.

Original post:
---
{base_content}
---

Respond in this JSON format:
{{
  "variations": [
    {{"title": "alt title", "content": "rewritten post text", "style": "story|tip|opinion|question"}}
  ]
}}""".strip()

        result = self._chat(
            model=self.text_model,
            messages=[
                {"role": "system", "content": "You are a LinkedIn content editor."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.9,
            max_tokens=1200,
        )

        try:
            parsed = json.loads(result)
            return parsed.get("variations", [])
        except json.JSONDecodeError:
            return [{"title": "Variation", "content": result}]

    def generate_image_prompts(self, post_content: str, count: int = 3) -> list[str]:
        """Generate detailed image prompts that would complement a LinkedIn post."""
        prompt = f"""\
Given this LinkedIn post, generate {count} detailed prompts for creating images \
that would complement it when posted on LinkedIn.

Post:
---
{post_content}
---

Each prompt should be:
- Specific enough for an AI image generator (like DALL-E, Midjourney, or Stable Diffusion)
- Professional and appropriate for LinkedIn
- Visually distinct from each other
- Under 100 words each

Respond in this JSON format:
{{
  "prompts": [
    "prompt 1 text",
    "prompt 2 text",
    "prompt 3 text"
  ]
}}""".strip()

        result = self._chat(
            model=self.text_model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an AI image prompt engineer.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.8,
            max_tokens=600,
        )

        try:
            parsed = json.loads(result)
            return parsed.get("prompts", [])
        except json.JSONDecodeError:
            return [result]

    def improve_post(self, post_content: str) -> dict:
        """Suggest improvements to an existing post draft."""
        prompt = f"""\
Review this LinkedIn post draft and suggest 3-5 specific improvements:

Post:
---
{post_content}
---

For each suggestion, explain WHY it would improve engagement.

Also provide an improved version of the post with your suggestions applied.

Respond in this JSON format:
{{
  "score": 7,
  "suggestions": [
    {{"area": "hook", "current": "current first line", "suggestion": "better first line", "reason": "why it works better"}}
  ],
  "improved_post": "The full improved version of the post",
  "hashtags": ["suggested", "hashtags"]
}}""".strip()

        result = self._chat(
            model=self.text_model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a LinkedIn content optimization expert.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=1200,
        )

        return self._parse_json_response(result, "improved_post")

    # ------------------------------------------------------------------
    # Profile analysis
    # ------------------------------------------------------------------

    def analyze_profile_screenshot(self, image_path: str) -> dict:
        """
        Analyze a LinkedIn profile screenshot and provide actionable suggestions.

        Args:
            image_path: Path to a screenshot of the LinkedIn profile

        Returns dict with 'overall_score', 'strengths', 'improvements', and 'specific_fixes'.
        """
        if not Path(image_path).exists():
            return {"error": f"File not found: {image_path}"}

        b64_image = encode_image(image_path)
        mime_type = get_image_extension(image_path)

        prompt = (
            "You are an expert LinkedIn profile reviewer. Analyze this LinkedIn profile screenshot "
            "with extreme attention to detail and provide actionable feedback.\n\n"
            "Evaluate these areas in order:\n"
            "1. **Profile Photo** — Professional? Clear? Appropriate background?\n"
            "2. **Headline** — Clear value proposition? Keywords? Specificity?\n"
            "3. **About Section** — Compelling first 3 lines? Shows personality/expertise? Call to action?\n"
            "4. **Experience** — Quantified achievements? Action verbs? Relevance?\n"
            "5. **Skills & Endorsements** — Top 3 skills relevant to stated role?\n"
            "6. **Overall Visual Layout** — Clean? Easy to scan? Professional appearance?\n"
            "7. **Engagement Signals** — Any visible post activity, recommendations, or profile views?\n\n"
            "For each section, give a score from 1-10 and a specific, actionable improvement suggestion.\n\n"
            "Respond in this exact JSON format:\n"
            "{\n"
            '  "overall_score": 7.5,\n'
            '  "sections": {\n'
            '    "profile_photo": {"score": 8, "note": "what you see", "suggestion": "specific fix"},\n'
            '    "headline": {"score": 6, "note": "what you see", "suggestion": "specific fix"},\n'
            '    "about": {"score": 5, "note": "what you see", "suggestion": "specific fix"},\n'
            '    "experience": {"score": 7, "note": "what you see", "suggestion": "specific fix"},\n'
            '    "skills": {"score": 6, "note": "what you see", "suggestion": "specific fix"},\n'
            '    "layout": {"score": 8, "note": "what you see", "suggestion": "specific fix"},\n'
            '    "engagement": {"score": 4, "note": "what you see", "suggestion": "specific fix"}\n'
            "  },\n"
            '  "top_priorities": [\n'
            '    "Priority 1 with specific action",\n'
            '    "Priority 2 with specific action",\n'
            '    "Priority 3 with specific action"\n'
            "  ],\n"
            '  "quick_wins": [\n'
            '    "Something they can fix in under 5 minutes",\n'
            '    "Another quick fix"\n'
            "  ]\n"
            "}"
        ).strip()

        result = self._chat(
            model=self.vision_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{b64_image}"
                            },
                        },
                    ],
                }
            ],
            temperature=0.3,
            max_tokens=1500,
        )

        return self._parse_json_response(result, "profile_analysis")

    def generate_image(self, prompt: str, save_path: str = None) -> dict:
        """
        Generate an image from a text prompt using NIM's image generation.

        Note: As of 2026, NIM's primary offering is LLMs (Llama). For image generation,
        you'll need to use a separate service. This method attempts to use NIM's
        interface if available, otherwise guides you to alternatives.

        Returns dict with 'success', 'prompt', 'save_path', and 'note'.
        """
        # Check if there's an image generation endpoint in NIM
        try:
            response = self.client.images.generate(
                model=self.image_model,
                prompt=prompt,
                n=1,
                size="1024x1024",
            )
            image_url = response.data[0].url

            if save_path:
                import urllib.request

                urllib.request.urlretrieve(image_url, save_path)
                return {
                    "success": True,
                    "prompt": prompt,
                    "save_path": save_path,
                    "url": image_url,
                }
            return {"success": True, "prompt": prompt, "url": image_url}

        except Exception as e:
            return {
                "success": False,
                "prompt": prompt,
                "error": str(e),
                "note": (
                    "NIM image generation may need a specific image model endpoint. "
                    "Alternative — copy this prompt into one of these:\n"
                    " • Leonardo AI (free tier available)\n"
                    " • Playground AI (free daily credits)\n"
                    " • DALL-E / Bing Image Creator (free)\n"
                    " • Midjourney (paid)\n"
                    f"Prompt to use: {prompt}"
                ),
            }

    # ------------------------------------------------------------------
    # JSON parsing helpers
    # ------------------------------------------------------------------

    def _parse_post_response(self, raw: str) -> Dict[str, Any]:
        """Parse AI response into structured post format with guaranteed keys."""
        parsed: Dict[str, Any] = {}
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code blocks
            if "```json" in raw:
                json_str = raw.split("```json")[1].split("```")[0].strip()
                try:
                    parsed = json.loads(json_str)
                except json.JSONDecodeError:
                    pass
            elif "```" in raw:
                try:
                    json_str = raw.split("```")[1].split("```")[0].strip()
                    parsed = json.loads(json_str)
                except (json.JSONDecodeError, IndexError):
                    pass

        # Normalize and ensure key presence
        hashtags = parsed.get("hashtags")
        if not isinstance(hashtags, list):
            hashtags = []
        else:
            hashtags = [str(x).strip() for x in hashtags if x]

        image_prompts = parsed.get("image_prompts")
        if not isinstance(image_prompts, list):
            image_prompts = []
        else:
            image_prompts = [str(x).strip() for x in image_prompts if x]

        return {
            "title": str(parsed.get("title") or "AI Generated Post").strip()[:100],
            "content": str(parsed.get("content") or raw).strip(),
            "hashtags": hashtags,
            "image_prompts": image_prompts,
        }

    def _parse_json_response(self, raw: str, key: str) -> Dict[str, Any]:
        """Parse AI response, handling markdown-wrapped JSON."""
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            if "```json" in raw:
                json_str = raw.split("```json")[1].split("```")[0].strip()
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    pass
            if "```" in raw:
                try:
                    json_str = raw.split("```")[1].split("```")[0].strip()
                    return json.loads(json_str)
                except (json.JSONDecodeError, IndexError):
                    pass
            return {"raw_response": raw, "error": "Could not parse JSON"}
