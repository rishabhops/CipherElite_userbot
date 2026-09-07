# =============================================================================
#  CipherElite Userbot Plugin - Cipher Agent
#
#  Plugin Name:    agent
#  Version:        1.0.0
#  Author:         CipherElite Dev (@rishabhops)
#  Repository:     https://github.com/rishabhops/CipherElite
#
#  LICENSE:        MIT
#
#  WHAT THIS IS:
#   An agentic layer on top of Cipher AI: instead of just answering questions,
#   `.agent <task>` lets Gemini decide to actually DO things on your account
#   (search chats, send a message, join/leave a group, delete or forward a
#   message) using Gemini's native function-calling / tool-use.
#
#  SAFETY MODEL (important — read before wiring this into a group the bot
#  reads messages from):
#   Tools are split into two tiers:
#     • SAFE  (read-only, current-chat-only) — run immediately, no confirm.
#     • RISKY (touches another chat, a group's membership, or deletes/
#       forwards anything) — the agent is NOT allowed to execute these
#       directly. It queues the action and replies with what it wants to
#       do; you must explicitly send `.confirm <id>` within 2 minutes or
#       it expires untouched. `.cancel <id>` discards it immediately.
#   This exists specifically to blunt prompt-injection: if the agent is
#   ever pointed at content written by someone else (a group message, a
#   forwarded doc, a web page), that content cannot make it silently DM
#   people, leave/join groups, or delete/forward things using your real
#   account — a human (you) always has to approve the risky step.
# =============================================================================

VERSION = "1.0.0"
CATEGORY = "cipher_ai"

import asyncio
import secrets
from datetime import datetime, timedelta
from google import genai
from google.genai import types
from telethon import events
from utils.utils import CipherElite
from utils.decorators import rishabh
from plugins.bot import add_handler

MAX_AGENT_STEPS = 5
CONFIRM_TIMEOUT_SECONDS = 120
INTENT_TIMEOUT_SECONDS = 120
AGENT_MODEL = "gemini-2.5-flash"
INTENT_MODEL = "gemini-flash-lite-latest"

# action_id -> {"name": str, "args": dict, "chat_id": int, "expires": datetime, "description": str}
PENDING_ACTIONS = {}

# chat_id -> {"task": str, "expires": datetime}  — waiting on a natural "haan/nahi" reply
PENDING_INTENT = {}

# message ids this plugin itself sent, so the conversational listener never reacts to its own output
BOT_SENT_IDS = set()

YES_WORDS = {"yes", "yeah", "yep", "yup", "ha", "haan", "han", "ok", "okay", "sure", "go", "proceed", "👍", "✅"}
NO_WORDS = {"no", "nahi", "na", "nope", "cancel", "stop", "don't", "dont"}
YES_PHRASES = ("kar do", "kardo", "kro", "karo", "go ahead", "proceed", "haan kar")
NO_PHRASES = ("mat karo", "matt karo", "rehne do", "chhodo", "cancel kar")

AGENT_SYSTEM_PROMPT = """You are Cipher Agent, an autonomous assistant with tool access on the user's own \
Telegram account (CipherElite Userbot). You can search/read the current chat freely. Actions that touch \
another chat, a group's membership, or delete/forward anything are NOT executed by you directly — calling \
those tools only QUEUES them for the human owner to confirm. Always explain in plain text what you did or \
queued, in 1-3 sentences. Only call a tool when it's actually needed to complete the task — don't call tools \
for questions that are answerable directly. Never claim an action succeeded unless the tool result says so."""


# =============================================================================
#  Tool schemas (Gemini function-calling)
# =============================================================================
SAFE_TOOL_DECLS = [
    {
        "name": "search_messages",
        "description": "Search recent messages in the CURRENT chat for text matching a query.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "query": {"type": "STRING", "description": "Text to search for"},
                "limit": {"type": "INTEGER", "description": "Max results, default 20, max 50"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_recent_messages",
        "description": "Fetch the most recent messages from the CURRENT chat (no search filter).",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "limit": {"type": "INTEGER", "description": "How many recent messages, default 20, max 50"},
            },
        },
    },
]

RISKY_TOOL_DECLS = [
    {
        "name": "send_message_to_chat",
        "description": "Queue sending a text message to a DIFFERENT chat/user than the current one. Requires human confirmation.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "chat": {"type": "STRING", "description": "Username (e.g. @someone), phone, or numeric chat ID of the target"},
                "text": {"type": "STRING", "description": "Message text to send"},
            },
            "required": ["chat", "text"],
        },
    },
    {
        "name": "join_group",
        "description": "Queue joining a group/channel by username or invite link. Requires human confirmation.",
        "parameters": {
            "type": "OBJECT",
            "properties": {"chat": {"type": "STRING", "description": "Username or invite link of the group/channel"}},
            "required": ["chat"],
        },
    },
    {
        "name": "leave_group",
        "description": "Queue leaving a group/channel. Requires human confirmation.",
        "parameters": {
            "type": "OBJECT",
            "properties": {"chat": {"type": "STRING", "description": "Username or numeric ID of the group/channel"}},
            "required": ["chat"],
        },
    },
    {
        "name": "delete_messages",
        "description": "Queue deleting specific messages by ID from a chat. Requires human confirmation.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "chat": {"type": "STRING", "description": "Chat where the messages live ('current' for this chat)"},
                "message_ids": {"type": "ARRAY", "items": {"type": "INTEGER"}, "description": "Message IDs to delete"},
            },
            "required": ["chat", "message_ids"],
        },
    },
    {
        "name": "forward_message",
        "description": "Queue forwarding one message from one chat to another. Requires human confirmation.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "from_chat": {"type": "STRING", "description": "Source chat ('current' for this chat)"},
                "message_id": {"type": "INTEGER", "description": "ID of the message to forward"},
                "to_chat": {"type": "STRING", "description": "Destination chat username or ID"},
            },
            "required": ["from_chat", "message_id", "to_chat"],
        },
    },
]

RISKY_TOOL_NAMES = {d["name"] for d in RISKY_TOOL_DECLS}
ALL_TOOLS = [types.Tool(function_declarations=SAFE_TOOL_DECLS + RISKY_TOOL_DECLS)]


def _describe_action(name, args):
    if name == "send_message_to_chat":
        return f"📤 Send message to `{args.get('chat')}`:\n\"{str(args.get('text',''))[:200]}\""
    if name == "join_group":
        return f"➕ Join group/channel `{args.get('chat')}`"
    if name == "leave_group":
        return f"➖ Leave group/channel `{args.get('chat')}`"
    if name == "delete_messages":
        return f"🗑 Delete message IDs `{args.get('message_ids')}` in `{args.get('chat')}`"
    if name == "forward_message":
        return f"↪️ Forward message `{args.get('message_id')}` from `{args.get('from_chat')}` to `{args.get('to_chat')}`"
    return f"{name}({args})"


def _classify_reply(text: str) -> str:
    """Cheap keyword-based classifier: 'yes' / 'no' / 'clarify' (no API call needed)."""
    low = text.strip().lower()
    if low in YES_WORDS or any(p in low for p in YES_PHRASES):
        return "yes"
    if low in NO_WORDS or any(p in low for p in NO_PHRASES):
        return "no"
    return "clarify"


def init(client):
    """Initialize the Cipher Agent plugin"""
    try:
        from plugins.ai_setup import ai_config
    except ImportError:
        print("❌ ERROR: ai_setup.py not found! Cipher Agent needs it for the Gemini key.")
        return False

    commands = [
        ".agent <task>     — Give the agent a task (or just type 'cipher <task>' naturally)",
        ".confirm <id>     — Approve a queued risky action",
        ".cancel <id>      — Discard a queued risky action",
        ".pending          — List actions currently waiting on your confirmation",
    ]
    add_handler("agent", commands, "Cipher Agent — tool-using AI with confirm-before-act safety on risky actions")

    async def _send_tracked(event, text, **kw):
        """Send a reply and remember its ID so our own conversational listener ignores it."""
        msg = await event.reply(text, **kw)
        BOT_SENT_IDS.add(msg.id)
        if len(BOT_SENT_IDS) > 1000:  # keep the set from growing forever
            for old_id in list(BOT_SENT_IDS)[:500]:
                BOT_SENT_IDS.discard(old_id)
        return msg

    async def understand_intent(task_text):
        """Ask Gemini to restate the task in plain language WITHOUT doing it — this is
        what gets shown to the user before anything actually runs."""
        api_key = ai_config.get_api_key()
        if not api_key:
            return "⚠️ API key set nahi hai — pehle `.setai <key>` chalao."
        try:
            gclient = genai.Client(api_key=api_key)
            prompt = (
                "In 1-2 short sentences, restate what task you understand needs to be done, "
                "matching the user's language/tone (Hinglish is fine). Do NOT perform it, "
                "do NOT call any tools — just describe your understanding plainly.\n\n"
                f"User's request: {task_text}"
            )
            resp = await gclient.aio.models.generate_content(
                model=INTENT_MODEL,
                contents=[types.Content(role="user", parts=[types.Part(text=prompt)])],
                config=types.GenerateContentConfig(temperature=0.3, max_output_tokens=150),
            )
            return (resp.text or task_text).strip()
        except Exception:
            return task_text  # fall back to echoing the raw task if the quick call fails

    async def start_intent_flow(event, task):
        understood = await understand_intent(task)
        PENDING_INTENT[event.chat_id] = {
            "task": task,
            "expires": datetime.now() + timedelta(seconds=INTENT_TIMEOUT_SECONDS),
        }
        await _send_tracked(event, f"Ok!\n\n{understood}\n\n**Karu?** (haan/nahi, ya bata do kya alag chahiye)")

    # -------------------------------------------------------------------
    # Safe tool execution (runs immediately, current chat only)
    # -------------------------------------------------------------------
    async def _exec_safe_tool(event, name, args):
        try:
            if name == "search_messages":
                limit = min(int(args.get("limit", 20) or 20), 50)
                query = args.get("query", "")
                results = []
                async for msg in event.client.iter_messages(event.chat_id, search=query, limit=limit):
                    if msg.text:
                        sender = await msg.get_sender()
                        sname = getattr(sender, "first_name", None) or getattr(sender, "title", "Unknown")
                        results.append({"id": msg.id, "sender": sname, "text": msg.text[:200], "date": str(msg.date)})
                return {"status": "ok", "count": len(results), "messages": results}

            if name == "get_recent_messages":
                limit = min(int(args.get("limit", 20) or 20), 50)
                results = []
                async for msg in event.client.iter_messages(event.chat_id, limit=limit):
                    if msg.text:
                        sender = await msg.get_sender()
                        sname = getattr(sender, "first_name", None) or getattr(sender, "title", "Unknown")
                        results.append({"id": msg.id, "sender": sname, "text": msg.text[:200], "date": str(msg.date)})
                return {"status": "ok", "count": len(results), "messages": results}

            return {"status": "error", "message": f"Unknown safe tool: {name}"}
        except Exception as e:
            return {"status": "error", "message": str(e)[:200]}

    def _queue_risky_tool(chat_id, name, args):
        action_id = secrets.token_hex(3)
        PENDING_ACTIONS[action_id] = {
            "name": name,
            "args": args,
            "chat_id": chat_id,
            "expires": datetime.now() + timedelta(seconds=CONFIRM_TIMEOUT_SECONDS),
            "description": _describe_action(name, args),
        }
        return action_id

    # -------------------------------------------------------------------
    # Risky tool execution (only runs from .confirm, never from the agent loop)
    # -------------------------------------------------------------------
    async def _exec_risky_action(event, name, args):
        try:
            if name == "send_message_to_chat":
                entity = await event.client.get_entity(args["chat"])
                await event.client.send_message(entity, args["text"])
                return "✅ Message sent."

            if name == "join_group":
                from telethon.tl.functions.channels import JoinChannelRequest
                entity = await event.client.get_entity(args["chat"])
                await event.client(JoinChannelRequest(entity))
                return "✅ Joined."

            if name == "leave_group":
                from telethon.tl.functions.channels import LeaveChannelRequest
                entity = await event.client.get_entity(args["chat"])
                await event.client(LeaveChannelRequest(entity))
                return "✅ Left."

            if name == "delete_messages":
                target = event.chat_id if args["chat"] == "current" else await event.client.get_entity(args["chat"])
                await event.client.delete_messages(target, args["message_ids"])
                return f"✅ Deleted {len(args['message_ids'])} message(s)."

            if name == "forward_message":
                src = event.chat_id if args["from_chat"] == "current" else await event.client.get_entity(args["from_chat"])
                dst = await event.client.get_entity(args["to_chat"])
                msg = await event.client.get_messages(src, ids=args["message_id"])
                if not msg:
                    return "❌ Source message not found."
                await event.client.forward_messages(dst, msg)
                return "✅ Forwarded."

            return f"❌ Unknown action: {name}"
        except Exception as e:
            return f"❌ Failed: {str(e)[:200]}"

    # -------------------------------------------------------------------
    # Agent loop
    # -------------------------------------------------------------------
    async def run_agent(event, task):
        api_key = ai_config.get_api_key()
        if not api_key:
            return "❌ **API Key not configured.** Use `.setai <key>` first."

        gclient = genai.Client(api_key=api_key)
        contents = [types.Content(role="user", parts=[types.Part(text=task)])]
        queued_this_run = []

        for _step in range(MAX_AGENT_STEPS):
            response = await gclient.aio.models.generate_content(
                model=AGENT_MODEL,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=AGENT_SYSTEM_PROMPT,
                    tools=ALL_TOOLS,
                    temperature=0.3,
                    max_output_tokens=1200,
                ),
            )
            candidate = response.candidates[0]
            parts = candidate.content.parts or []
            calls = [p.function_call for p in parts if getattr(p, "function_call", None)]

            if not calls:
                text = response.text or "🤖 (no response)"
                if queued_this_run:
                    ids = ", ".join(f"`{i}`" for i in queued_this_run)
                    text += f"\n\n⏳ **Waiting on your confirmation:** {ids}\nUse `.confirm <id>` or `.cancel <id>` (expires in 2 min)."
                return text

            contents.append(candidate.content)
            response_parts = []
            stop_after_this = False

            for fc in calls:
                name = fc.name
                args = dict(fc.args) if fc.args else {}

                if name in RISKY_TOOL_NAMES:
                    action_id = _queue_risky_tool(event.chat_id, name, args)
                    queued_this_run.append(action_id)
                    result = {"status": "queued_for_human_confirmation", "action_id": action_id}
                    stop_after_this = True  # don't let the model chain further actions blind
                else:
                    result = await _exec_safe_tool(event, name, args)

                response_parts.append(types.Part.from_function_response(name=name, response=result))

            contents.append(types.Content(role="user", parts=response_parts))

            if stop_after_this:
                # Ask the model for a final summary now, without letting it queue more actions
                wrap_up = await gclient.aio.models.generate_content(
                    model=AGENT_MODEL,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=AGENT_SYSTEM_PROMPT,
                        temperature=0.3,
                        max_output_tokens=500,
                    ),
                )
                text = wrap_up.text or "🤖 Action(s) queued."
                ids = ", ".join(f"`{i}`" for i in queued_this_run)
                text += f"\n\n⏳ **Waiting on your confirmation:** {ids}\nUse `.confirm <id>` or `.cancel <id>` (expires in 2 min)."
                return text

        return "⚠️ Agent stopped — too many steps for one task. Try breaking it into smaller requests."

    # -------------------------------------------------------------------
    # Handlers
    # -------------------------------------------------------------------
    @CipherElite.on(events.NewMessage(pattern=r"\.agent(?:\s+(.*))?"))
    @rishabh()
    async def agent_handler(event):
        try:
            task = event.pattern_match.group(1)
            if not task:
                await event.reply(
                    "🤖 **Cipher Agent**\n\n"
                    "Usage: `.agent <task>` — or just type `cipher <task>` naturally.\n\n"
                    "Examples:\n"
                    "`.agent find messages mentioning 'invoice' in this chat`\n"
                    "`cipher iss chat ke 10 recent messages padho aur summary do`\n\n"
                    "Main pehle bataunga ki maine kya samjha, phir puchunga \"Karu?\" — "
                    "tumhare haan/confirm ke baad hi kuch execute hoga. Risky actions "
                    "(dusri chat me message, group join/leave, delete/forward) ke liye ek "
                    "extra `.confirm <id>` bhi lagega."
                )
                return
            await start_intent_flow(event, task)
        except Exception as e:
            print(f"❌ agent_handler error: {e}")
            try:
                await event.reply(f"❌ **Agent error:** {str(e)[:200]}")
            except Exception:
                pass

    @CipherElite.on(events.NewMessage(pattern=r"(?i)^cipher[,:]?\s+(.+)"))
    @rishabh()
    async def cipher_trigger_handler(event):
        try:
            existing = PENDING_INTENT.get(event.chat_id)
            if existing and datetime.now() <= existing["expires"]:
                return  # a live prompt is already waiting on a reply — don't stack a second one
            task = event.pattern_match.group(1)
            await start_intent_flow(event, task)
        except Exception as e:
            print(f"❌ cipher_trigger_handler error: {e}")
            try:
                await event.reply(f"❌ **Agent error:** {str(e)[:200]}")
            except Exception:
                pass

    @CipherElite.on(events.NewMessage())
    @rishabh()
    async def intent_reply_handler(event):
        """Listens for the plain 'haan/nahi/kuch aur' reply to a pending 'Karu?' prompt."""
        try:
            if event.id in BOT_SENT_IDS:
                return
            text = (event.raw_text or "").strip()
            if not text or text.startswith("."):
                return  # real commands are handled by their own dedicated handlers

            chat_id = event.chat_id
            pending = PENDING_INTENT.get(chat_id)
            if not pending:
                return
            if datetime.now() > pending["expires"]:
                del PENDING_INTENT[chat_id]
                return

            verdict = _classify_reply(text)

            if verdict == "yes":
                del PENDING_INTENT[chat_id]
                thinking = await _send_tracked(event, "🤖 **Agent working...**")
                try:
                    result = await asyncio.wait_for(run_agent(event, pending["task"]), timeout=60.0)
                except asyncio.TimeoutError:
                    result = "⏰ **Timeout** — try a simpler task."
                except Exception as e:
                    result = f"❌ **Agent error:** {str(e)[:200]}"
                await thinking.edit(result[:4000])
                return

            if verdict == "no":
                del PENDING_INTENT[chat_id]
                await _send_tracked(event, "Ok, cancel kar diya. 👍")
                return

            # Anything else = a clarification / correction — re-interpret and ask again
            understood = await understand_intent(text)
            PENDING_INTENT[chat_id] = {
                "task": text,
                "expires": datetime.now() + timedelta(seconds=INTENT_TIMEOUT_SECONDS),
            }
            await _send_tracked(event, f"Ok!\n\n{understood}\n\n**Karu?** (haan/nahi, ya bata do kya alag chahiye)")
        except Exception as e:
            print(f"❌ intent_reply_handler error: {e}")
            try:
                await event.reply(f"❌ **Agent error:** {str(e)[:200]}")
            except Exception:
                pass

    @CipherElite.on(events.NewMessage(pattern=r"\.confirm(?:\s+(\S+))?$"))
    @rishabh()
    async def confirm_handler(event):
        action_id = event.pattern_match.group(1)
        if not action_id:
            await event.reply("❌ Usage: `.confirm <id>` — see `.pending` for active IDs.")
            return
        entry = PENDING_ACTIONS.get(action_id)
        if not entry:
            await event.reply("❌ No pending action with that ID (it may have expired or already run).")
            return
        if datetime.now() > entry["expires"]:
            del PENDING_ACTIONS[action_id]
            await event.reply("⌛ That action expired. Ask the agent again if you still want it done.")
            return

        msg = await event.reply(f"⏳ Executing: {entry['description']}...")
        result = await _exec_risky_action(event, entry["name"], entry["args"])
        del PENDING_ACTIONS[action_id]
        await msg.edit(f"{entry['description']}\n\n{result}")

    @CipherElite.on(events.NewMessage(pattern=r"\.cancel(?:\s+(\S+))?$"))
    @rishabh()
    async def cancel_handler(event):
        action_id = event.pattern_match.group(1)
        if not action_id:
            await event.reply("❌ Usage: `.cancel <id>` — see `.pending` for active IDs.")
            return
        entry = PENDING_ACTIONS.pop(action_id, None)
        if not entry:
            await event.reply("❌ No pending action with that ID.")
            return
        await event.reply(f"🗑 Cancelled: {entry['description']}")

    @CipherElite.on(events.NewMessage(pattern=r"\.pending$"))
    @rishabh()
    async def pending_handler(event):
        now = datetime.now()
        active = {k: v for k, v in PENDING_ACTIONS.items() if v["expires"] > now}
        # sweep expired ones while we're here
        for k in list(PENDING_ACTIONS):
            if PENDING_ACTIONS[k]["expires"] <= now:
                del PENDING_ACTIONS[k]

        if not active:
            await event.reply("📭 **No pending actions.**")
            return
        lines = []
        for action_id, entry in active.items():
            secs_left = int((entry["expires"] - now).total_seconds())
            lines.append(f"`{action_id}` — {entry['description']} ({secs_left}s left)")
        await event.reply("⏳ **Pending confirmations:**\n\n" + "\n\n".join(lines))

    print(f"✅ Cipher Agent Plugin v{VERSION} initialized (safe tools auto-run, risky tools require .confirm)")
    return True
