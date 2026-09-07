# =============================================================================
#  CipherElite Userbot Plugin - Cipher AI (Google Gemini) v2.0
#  With Repository Data Access, Real Chat Memory, Vision, Model Switching
#  and Free/Paid API Key Auto-Fallback
#
#  Plugin Name:    cipher_ai
#  Author:         CipherElite Dev (@rishabhops)
#  Repository:     https://github.com/rishabhops/CipherElite
#
#  LICENSE:        MIT
#
#  CHANGELOG (v1.0.0 -> v2.0.0):
#   - Migrated google.generativeai (deprecated, EOL) -> google-genai SDK
#   - Uses client.aio (true async) instead of blocking sync calls, so the
#     bot no longer freezes while waiting on Gemini
#   - Automatic model fallback chain + retry-with-backoff so FREE-tier keys
#     that hit rate limits degrade gracefully instead of erroring out
#   - Paid-tier users can pin a stronger model (e.g. gemini-3.1-pro-preview) via
#     .aimodel, while still falling back to free models if that model
#     ever gets rate limited
#   - New commands: .aimodel, .aitemp, .aistats, .aiimg (vision support)
#   - .ai now auto-detects an image in the replied-to message and sends it
#     to Gemini for vision analysis alongside the text question
# =============================================================================

VERSION = "2.0.0"
CATEGORY = "utilities"

import asyncio
import time
import aiohttp
from google import genai
from google.genai import types
from telethon import events
from utils.utils import CipherElite
from utils.decorators import rishabh
from plugins.bot import add_handler
from vars import ELITE_BOT_USERNAME

# ---------------------------------------------------------------------------
# In-memory state (per chat_id)
# ---------------------------------------------------------------------------
conversation_history = {}   # chat_id -> [{"role": "user"/"assistant", "content": str}]
chat_settings = {}          # chat_id -> {"model": str, "temperature": float}
ai_stats = {}                # chat_id -> {"queries": int, "errors": int, "rate_limited": int}

# Models that reliably work on a FREE Gemini API key. Order = fallback order.
# NOTE: Google retired the Gemini 2.x/2.5 lineup for new API keys in 2026 —
# these must be the current Gemini 3.x model IDs, or every request 404s
# regardless of quota. "gemini-flash-lite-latest" is a rolling alias that
# always points at Google's current lite model, kept last as a safety net
# against the *next* rename too.
FREE_MODEL_CHAIN = [
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash-lite",
    "gemini-flash-lite-latest",
]

# Models a PAID/billing-enabled key can additionally use.
PRO_MODELS = ["gemini-3.1-pro-preview"]

ALL_SELECTABLE_MODELS = PRO_MODELS + FREE_MODEL_CHAIN
DEFAULT_MODEL = FREE_MODEL_CHAIN[0]

# Strings Gemini returns when a key/model is out of quota — used to decide
# when to retry / fall back instead of just failing.
RATE_LIMIT_MARKERS = ("429", "RESOURCE_EXHAUSTED", "QUOTA", "rate limit")

SYSTEM_PROMPT = """You are **Cipher AI**, a specialized AI assistant created for the **CipherElite Userbot**.

**ABOUT YOU (ONLY MENTION IF EXPLICITLY ASKED):**
• **Name:** Cipher AI
• **Created by:** Rishabh Anand (@rishabhops)
• **Owner/Creator's Telegram:** @thanosceo
• **Project:** CipherElite - Advanced Telegram Userbot
• **Repository:** https://github.com/rishabhops/CipherElite
• **Primary Repo Branch:** cooking

**YOUR PURPOSE:**
You are integrated into the CipherElite Telegram Userbot. Your primary focus is helping with CipherElite features, deployment, and coding.
HOWEVER, you are also a general-purpose AI. You MUST answer general everyday questions (like career advice, education, general knowledge, etc.) naturally and helpfully without restricting yourself to technical topics. You can also analyze images when the user replies to a photo with a command.

**PERSONALITY & BEHAVIOR:**
1. ONLY introduce yourself or mention your creators if the user EXPLICITLY asks questions like "who are you", "who made you", or "what is your name". Do NOT inject your identity into normal answers.
2. Answer whatever the user asks directly. Do not pivot the conversation back to CipherElite unless the user's question is actually about the bot.
3. Be helpful, concise, and professional. Act like a natural conversational partner.
4. Use **bold formatting** for important keywords.
5. For simple questions: Keep SHORT (1-2 paragraphs).
6. For complex questions: Provide COMPLETE detailed answers using bullet points and numbered lists.
7. Never apologize unnecessarily or add disclaimers about being an AI.
8. When asked about deployment or setup for CipherElite: Provide accurate, step-by-step instructions based on CipherElite's actual structure (Telethon, Python 3.8+, VPS deployment, SQLite databases).
"""


async def fetch_repository_data(owner="rishabhops", repo="CipherElite", branch="cooking"):
    """Fetch repository structure and README from GitHub"""
    try:
        async with aiohttp.ClientSession() as session:
            readme_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/README.md"
            async with session.get(readme_url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                readme_content = await resp.text() if resp.status == 200 else ""

            setup_urls = [
                f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/requirements.txt",
                f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/setup.md",
                f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/SETUP.md",
            ]
            setup_content = ""
            for url in setup_urls:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status == 200:
                        setup_content = await resp.text()
                        break

            return {
                # Trimmed from 2000->800 chars each: this context is re-sent on
                # every CipherElite-related query, so keeping it lean saves
                # input tokens across the whole conversation, not just once.
                "readme": readme_content[:800] if readme_content else "",
                "setup": setup_content[:800] if setup_content else "",
                "has_data": bool(readme_content or setup_content),
            }
    except Exception as e:
        print(f"⚠️ Failed to fetch repo data: {e}")
        return {"readme": "", "setup": "", "has_data": False}


def init(client):
    """Initialize the Cipher AI plugin"""
    try:
        from plugins.ai_setup import ai_config  # Centralized API key storage
    except ImportError:
        print("❌ ERROR: ai_setup.py not found! Please create it first.")
        return False

    commands = [
        ".ai <question>      — Ask Cipher AI a question (reply to a photo to analyze it)",
        ".aiimg <question>   — Force image analysis on the replied-to photo",
        ".aiclear            — Clear conversation history",
        ".aimodel [name]     — View/switch/pin the AI model for this chat",
        ".aitemp [0-2]       — View/set response creativity (temperature)",
        ".aihistory [n]      — View/set how many past messages Cipher AI remembers (token usage)",
        ".aistats            — Show usage stats for this chat",
        ".aiinfo             — Show AI info",
    ]
    add_handler("cipher_ai", commands, "Cipher AI v2.0 - Google Gemini, free/paid auto-fallback + vision")

    # -------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------
    def get_settings(chat_id):
        if chat_id not in chat_settings:
            chat_settings[chat_id] = {
                "model": DEFAULT_MODEL,
                "temperature": 0.7,
                "pinned": False,       # True once user manually runs .aimodel
                "history_limit": 12,   # messages kept (6 exchanges) — lower = fewer tokens/request
            }
        return chat_settings[chat_id]

    def pick_model_for_query(chat_id, response_type):
        """
        Token-saving smart routing: if the user hasn't manually pinned a model,
        route short/medium questions to Flash-Lite (cheapest, fewest tokens)
        and only use full Flash for questions that actually need depth
        (detailed / CipherElite / technical). A manual .aimodel pin always wins.
        """
        settings = get_settings(chat_id)
        if settings["pinned"]:
            return settings["model"]
        if response_type == "detailed":
            return "gemini-3.7-flash"
        return "gemini-flash-lite-latest"

    def max_tokens_for_type(response_type):
        """Cap output length by question type so simple answers don't burn a full 2000-token budget."""
        return {"short": 400, "medium": 900, "detailed": 2000}.get(response_type, 900)

    def bump_stat(chat_id, key):
        if chat_id not in ai_stats:
            ai_stats[chat_id] = {"queries": 0, "errors": 0, "rate_limited": 0}
        ai_stats[chat_id][key] = ai_stats[chat_id].get(key, 0) + 1

    def is_rate_limit_error(err_text: str) -> bool:
        low = err_text.lower()
        return any(marker.lower() in low for marker in RATE_LIMIT_MARKERS)

    async def call_gemini_once(api_key, contents, system_instruction, model_name, temperature, max_output_tokens=2000):
        """Single non-blocking call to Gemini via the async client. Raises on error."""
        gclient = genai.Client(api_key=api_key)
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            top_p=0.9,
        )
        response = await gclient.aio.models.generate_content(
            model=model_name,
            contents=contents,
            config=config,
        )
        return response.text

    async def make_ai_request(chat_id, messages, repo_context="", image_bytes=None, image_mime=None,
                               preferred_model=None, max_output_tokens=900):
        """
        Builds the request, tries the preferred model first, and on
        rate-limit errors retries with a short backoff, then falls through
        the free-tier model chain so a maxed-out key still gets an answer.
        Returns (model_used, text) on success, or (None, error_message) on failure.
        """
        api_key = ai_config.get_api_key()
        if not api_key:
            return None, (
                "❌ **API Key not configured!**\n\n"
                "Use `.setai <key>` to set up Google Gemini API.\n\n"
                "🔗 Get a **free** key: https://aistudio.google.com/"
            )

        settings = get_settings(chat_id)
        first_choice = preferred_model or settings["model"]
        enhanced_prompt = SYSTEM_PROMPT
        if repo_context:
            enhanced_prompt += f"\n\n**CURRENT REPOSITORY CONTEXT:**\n{repo_context}"

        # Build Gemini-native content list
        gemini_contents = []
        for msg in messages:
            role = "user" if msg["role"] == "user" else "model"
            gemini_contents.append(types.Content(role=role, parts=[types.Part(text=msg["content"])]))

        if image_bytes and gemini_contents:
            gemini_contents[-1].parts.append(
                types.Part.from_bytes(data=image_bytes, mime_type=image_mime or "image/jpeg")
            )

        # Preferred model first, then the free fallback chain (deduped, order kept)
        candidates = [first_choice] + [m for m in FREE_MODEL_CHAIN if m != first_choice]

        last_error = "Unknown error"
        for model_name in candidates:
            for attempt in range(2):  # 1 retry per model on rate-limit before moving on
                try:
                    text = await call_gemini_once(
                        api_key, gemini_contents, enhanced_prompt, model_name,
                        settings["temperature"], max_output_tokens,
                    )
                    return model_name, text
                except Exception as e:
                    err_text = str(e)
                    last_error = err_text
                    if is_rate_limit_error(err_text):
                        bump_stat(chat_id, "rate_limited")
                        if attempt == 0:
                            await asyncio.sleep(3)  # brief backoff, then retry same model once
                            continue
                        break  # give up on this model, try next in chain
                    elif "404" in err_text or "NOT_FOUND" in err_text or "no longer available" in err_text.lower():
                        # This specific model was retired/renamed by Google — skip it
                        # immediately and try the next one in the fallback chain.
                        break
                    else:
                        # Non-rate-limit error (bad key, safety block, etc.) - stop immediately
                        return None, f"❌ **Error:** {err_text[:150]}"

        return None, (
            f"❌ **All models are rate-limited right now.**\n"
            f"Free-tier quota resets after a short cooldown — try again in a minute.\n"
            f"`{str(last_error)[:100]}`"
        )

    def estimate_response_type(query):
        short_keywords = ["what is", "who is", "when", "where", "how many", "define", "meaning", "your name", "who made", "who created"]
        complex_keywords = ["how to", "deploy", "setup", "install", "tutorial", "guide", "step", "process", "configure", "build", "create", "write code", "cipherelite", "userbot"]
        query_lower = query.lower()
        if any(k in query_lower for k in complex_keywords):
            return "detailed"
        if any(k in query_lower for k in short_keywords):
            return "short"
        return "medium"

    def format_response(response, response_type):
        response = response.strip()
        unnecessary_phrases = [
            "I'm glad you asked. ", "Great question! ", "Thank you for asking. ",
            "Let me explain: ", "Sure, here's ",
        ]
        for phrase in unnecessary_phrases:
            if response.startswith(phrase):
                response = response[len(phrase):]
        response = response.strip()
        if response_type == "short" and len(response) > 500:
            sentences = response.split(". ")
            response = ". ".join(sentences[:3]) + "."
        elif response_type == "detailed" and len(response) > 3500:
            response = response + "\n\n💡 *Response may continue in next message if too long*"
        return response

    async def download_reply_image(event):
        """If the message this command replies to has a photo, download it."""
        if not event.is_reply:
            return None, None
        reply_msg = await event.get_reply_message()
        if not reply_msg or not reply_msg.photo:
            return None, None
        img_bytes = await event.client.download_media(reply_msg, bytes)
        return img_bytes, "image/jpeg"

    async def send_long_message(thinking_msg, event, formatted):
        """Handles Telegram's 4096 char limit by splitting on paragraph breaks."""
        if len(formatted) <= 4096:
            await thinking_msg.edit(formatted)
            return
        messages = []
        current_msg = ""
        for part in formatted.split("\n\n"):
            if len(current_msg) + len(part) + 4 > 4000:
                if current_msg:
                    messages.append(current_msg.strip())
                current_msg = part
            else:
                current_msg += "\n\n" + part if current_msg else part
        if current_msg.strip():
            messages.append(current_msg.strip())
        if messages:
            await thinking_msg.edit(messages[0])
            for msg in messages[1:]:
                await asyncio.sleep(0.5)
                await event.reply(msg)

    async def run_ai_query(event, query, force_image=False):
        """Shared logic used by .ai and .aiimg"""
        if not ai_config.is_enabled():
            await event.reply(
                "❌ **API Key Not Set**\n\nUse `.setai <your_gemini_api_key>`\n\n"
                "🔗 Get a **free** key: https://aistudio.google.com/"
            )
            return
        if len(query) > 2000:
            await event.reply("📝 **Query too long!** Max 2000 characters.")
            return

        thinking_msg = await event.reply("🤔 **Cipher AI thinking...**")
        chat_id = event.chat_id

        image_bytes, image_mime = await download_reply_image(event)
        if force_image and not image_bytes:
            await thinking_msg.edit("❌ **Reply to a photo** to use `.aiimg`.")
            return
        if image_bytes:
            await thinking_msg.edit("🤔 **Cipher AI thinking...** (analyzing image...)")

        response_type = estimate_response_type(query)

        repo_context = ""
        cipher_keywords = ["cipherelite", "cipher elite", "userbot setup", "userbot deploy", "this bot's repo"]
        if any(k in query.lower() for k in cipher_keywords):
            await thinking_msg.edit("🤔 **Cipher AI thinking...** (scanning repository...)")
            repo_data = await fetch_repository_data()
            if repo_data["has_data"]:
                repo_context = f"README excerpt:\n{repo_data['readme']}\n\nSetup guide:\n{repo_data['setup']}"

        settings = get_settings(chat_id)
        history_limit = settings["history_limit"]

        if chat_id not in conversation_history:
            conversation_history[chat_id] = []
        conversation_history[chat_id].append({"role": "user", "content": query})
        if len(conversation_history[chat_id]) > history_limit:
            conversation_history[chat_id] = conversation_history[chat_id][-history_limit:]
            if conversation_history[chat_id][0]["role"] == "assistant":
                conversation_history[chat_id] = conversation_history[chat_id][1:]

        routed_model = pick_model_for_query(chat_id, response_type)
        token_cap = max_tokens_for_type(response_type)

        try:
            model_used, response = await asyncio.wait_for(
                make_ai_request(
                    chat_id, conversation_history[chat_id], repo_context, image_bytes, image_mime,
                    preferred_model=routed_model, max_output_tokens=token_cap,
                ),
                timeout=45.0,
            )
        except asyncio.TimeoutError:
            model_used, response = None, "⏰ **Timeout:** Request took too long. Try again or use `.aiclear` to reset."

        bump_stat(chat_id, "queries")

        if model_used is None:
            bump_stat(chat_id, "errors")
            await thinking_msg.edit(response)
            if conversation_history[chat_id] and conversation_history[chat_id][-1]["role"] == "user":
                conversation_history[chat_id].pop()
            return

        response = format_response(response, response_type)
        conversation_history[chat_id].append({"role": "assistant", "content": response})

        model_tag = f"🔧 `{model_used}`" if model_used != routed_model else ""
        if response_type == "short":
            formatted = f"🤖 **Cipher AI:**\n\n{response}"
        elif response_type == "detailed":
            formatted = (
                f"🤖 **Detailed Response:** {model_tag}\n\n{response}\n\n"
                f"═════════════════════\n📌 **Q:** `{query[:60]}{'...' if len(query) > 60 else ''}`"
            )
        else:
            formatted = (
                f"🤖 **Cipher AI Response:** {model_tag}\n\n{response}\n\n"
                f"─────────────────\n💭 Q: `{query[:60]}{'...' if len(query) > 60 else ''}`"
            )

        await send_long_message(thinking_msg, event, formatted)

    # -------------------------------------------------------------------
    # Handlers
    # -------------------------------------------------------------------
    @CipherElite.on(events.NewMessage(pattern=r"\.ai(?:\s+(.*))?"))
    @rishabh()
    async def ai_handler(event):
        try:
            query = event.pattern_match.group(1)
            if not query:
                await event.reply(
                    "❓ **How to use Cipher AI:**\n\n"
                    "**About Me:**\n`.ai Who are you?`\n\n"
                    "**CipherElite Help:**\n`.ai How to deploy CipherElite?`\n\n"
                    "**General Questions:**\n`.ai What is Python?`\n\n"
                    "**Vision:** reply to a photo with `.ai describe this` or use `.aiimg`\n\n"
                    "**Other commands:** `.aimodel`, `.aitemp`, `.aistats`, `.aiclear`, `.aiinfo`"
                )
                return
            await run_ai_query(event, query)
        except Exception as e:
            print(f"❌ AI Handler Error: {e}")
            try:
                await event.reply(f"❌ **Error:** {str(e)[:100]}")
            except Exception:
                pass

    @CipherElite.on(events.NewMessage(pattern=r"\.aiimg(?:\s+(.*))?"))
    @rishabh()
    async def aiimg_handler(event):
        try:
            query = event.pattern_match.group(1) or "Describe this image in detail."
            await run_ai_query(event, query, force_image=True)
        except Exception as e:
            print(f"❌ AIIMG Handler Error: {e}")
            try:
                await event.reply(f"❌ **Error:** {str(e)[:100]}")
            except Exception:
                pass

    @CipherElite.on(events.NewMessage(pattern=r"\.aiclear"))
    @rishabh()
    async def aiclear_handler(event):
        chat_id = event.chat_id
        if chat_id in conversation_history:
            msg_count = len(conversation_history[chat_id])
            del conversation_history[chat_id]
            await event.reply(f"🗑️ **Cleared!** {msg_count} messages removed. My memory is now fresh.")
        else:
            await event.reply("📭 **No history** in this chat.")

    @CipherElite.on(events.NewMessage(pattern=r"\.aimodel(?:\s+(.*))?"))
    @rishabh()
    async def aimodel_handler(event):
        chat_id = event.chat_id
        settings = get_settings(chat_id)
        arg = event.pattern_match.group(1)
        if not arg:
            model_list = "\n".join(
                f"• `{m}`{' (needs paid/billing key)' if m in PRO_MODELS else ' (free-tier friendly)'}"
                for m in ALL_SELECTABLE_MODELS
            )
            pin_note = f"📌 **Pinned:** `{settings['model']}`" if settings["pinned"] else "🤖 **Auto-routing** (token-saver: short/medium questions → Flash-Lite, detailed → Flash)"
            await event.reply(
                f"{pin_note}\n\n"
                f"**Available models:**\n{model_list}\n\n"
                f"Usage: `.aimodel gemini-3.1-pro-preview` — pin a model\n"
                f"`.aimodel auto` — go back to token-saving auto-routing\n\n"
                f"ℹ️ Free API keys work fine with the flash/lite models. If a pinned model gets rate limited, "
                f"Cipher AI automatically falls back to a free model so you still get an answer."
            )
            return
        arg = arg.strip()
        if arg.lower() == "auto":
            settings["pinned"] = False
            await event.reply("✅ Back to **auto-routing** — Cipher AI will pick the cheapest model that fits each question.")
            return
        if arg not in ALL_SELECTABLE_MODELS:
            await event.reply(f"❌ Unknown model `{arg}`. Run `.aimodel` with no arguments to see the list.")
            return
        settings["model"] = arg
        settings["pinned"] = True
        await event.reply(f"✅ Model **pinned** to `{arg}` for this chat. Use `.aimodel auto` to unpin.")

    @CipherElite.on(events.NewMessage(pattern=r"\.aitemp(?:\s+(.*))?"))
    @rishabh()
    async def aitemp_handler(event):
        chat_id = event.chat_id
        settings = get_settings(chat_id)
        arg = event.pattern_match.group(1)
        if not arg:
            await event.reply(
                f"🌡️ **Current temperature:** `{settings['temperature']}`\n\n"
                f"Usage: `.aitemp 0.9` — lower = more precise/factual, higher = more creative (range 0.0–2.0)"
            )
            return
        try:
            val = float(arg.strip())
            if not (0.0 <= val <= 2.0):
                raise ValueError
        except ValueError:
            await event.reply("❌ Enter a number between `0.0` and `2.0`, e.g. `.aitemp 0.7`")
            return
        settings["temperature"] = val
        await event.reply(f"✅ Temperature for this chat set to `{val}`.")

    @CipherElite.on(events.NewMessage(pattern=r"\.aihistory(?:\s+(.*))?"))
    @rishabh()
    async def aihistory_handler(event):
        chat_id = event.chat_id
        settings = get_settings(chat_id)
        arg = event.pattern_match.group(1)
        if not arg:
            await event.reply(
                f"🧠 **Memory window:** `{settings['history_limit']}` messages (~{settings['history_limit'] // 2} exchanges)\n\n"
                f"Usage: `.aihistory 8` — lower = fewer tokens sent per request (cheaper, less context retained), "
                f"higher = remembers more of the conversation (range 2–30)."
            )
            return
        try:
            val = int(arg.strip())
            if not (2 <= val <= 30):
                raise ValueError
        except ValueError:
            await event.reply("❌ Enter a whole number between `2` and `30`, e.g. `.aihistory 8`")
            return
        settings["history_limit"] = val
        if chat_id in conversation_history and len(conversation_history[chat_id]) > val:
            conversation_history[chat_id] = conversation_history[chat_id][-val:]
        await event.reply(f"✅ Memory window set to `{val}` messages for this chat.")

    @CipherElite.on(events.NewMessage(pattern=r"\.aistats"))
    @rishabh()
    async def aistats_handler(event):
        chat_id = event.chat_id
        stats = ai_stats.get(chat_id, {"queries": 0, "errors": 0, "rate_limited": 0})
        settings = get_settings(chat_id)
        history_len = len(conversation_history.get(chat_id, []))
        await event.reply(
            "📊 **Cipher AI Stats (this chat)**\n\n"
            f"• Queries: `{stats.get('queries', 0)}`\n"
            f"• Errors: `{stats.get('errors', 0)}`\n"
            f"• Rate-limit hits: `{stats.get('rate_limited', 0)}`\n"
            f"• Current model: `{settings['model']}`\n"
            f"• Temperature: `{settings['temperature']}`\n"
            f"• History size: `{history_len}` messages"
        )

    @CipherElite.on(events.NewMessage(pattern=r"\.aiinfo"))
    @rishabh()
    async def aiinfo_handler(event):
        is_enabled = ai_config.is_enabled()
        status_emoji = "✅" if is_enabled else "❌"
        settings = get_settings(event.chat_id)

        info = f"""🤖 **Cipher AI v{VERSION} - About Me**

**Name:** Cipher AI
**Creator:** Rishabh Anand (@rishabhops)
**Owner:** @thanosceo
**Project:** CipherElite Userbot

{status_emoji} **Status:** {'Active' if is_enabled else 'Inactive'}
🔧 **Model (this chat):** `{settings['model']}`
🌐 **SDK:** google-genai (async, non-blocking)
💬 **Active Chats:** {len(conversation_history)}

📝 **Features:**
• Real Chat Memory (per chat, adjustable window)
• Vision — analyze photos (`.ai` / `.aiimg` on a reply)
• Free/paid key auto-fallback across models on rate limits
• Token-saving auto-routing (cheap model for simple Qs, Flash for detailed ones)
• Repository-aware answers for CipherElite questions

📚 **Commands:**
• `.ai <question>` — Ask me anything (reply to a photo to analyze it)
• `.aiimg [question]` — Force image analysis on the replied photo
• `.aiclear` — Clear memory in this chat
• `.aimodel [name|auto]` — View/pin/unpin model
• `.aitemp [0-2]` — View/set creativity
• `.aihistory [n]` — View/set memory window (token usage)
• `.aistats` — Usage stats for this chat
• `.aiinfo` — About me

🔗 **Links:**
• GitHub: https://github.com/rishabhops/CipherElite
• Creator: @rishabhops
• Owner: @thanosceo"""
        await event.reply(info)

    print(f"✅ Cipher AI Plugin v{VERSION} initialized (google-genai, async, vision, free/paid fallback)")
    return True
