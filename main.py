#!/usr/bin/env python3
"""
The Nagger - A Telegram bot that harasses you about your overdue tasks.
"""

import os
import json
import asyncio
import logging
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from typing import Optional
from collections import defaultdict

import psycopg2
import dateparser
from dotenv import load_dotenv
import google.generativeai as genai
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Bot token from environment
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# Gemini API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


# Database Connection
DATABASE_URL = os.getenv("DATABASE_URL")
conn = None

try:
    conn = psycopg2.connect(DATABASE_URL)
    conn.autocommit = True
except Exception as e:
    logger.error(f"Failed to connect to DB: {e}")

# Timezone configuration - set to UTC+8 (Singapore/Malaysia/Hong Kong)
# Override with TIMEZONE env var if needed (e.g., "America/New_York")
LOCAL_TZ = ZoneInfo(os.getenv("TIMEZONE", "Asia/Singapore"))

def get_now():
    """Get current time in configured timezone."""
    return datetime.now(LOCAL_TZ)

# Nag interval in minutes (don't spam every minute)
NAG_INTERVAL_MINUTES = 5

# Rate limiting: max LLM requests per user per hour
LLM_RATE_LIMIT_PER_HOUR = 20
llm_usage_tracker = defaultdict(list)  # {user_id: [timestamp1, timestamp2, ...]}

# Nag messages by level
NAG_MESSAGES = {
    0: "Hey, '{task}' is due. Maybe get on it?",
    1: "You said you'd do '{task}'. Do it.",
    2: "Seriously? '{task}' is STILL not done?",
    3: "'{task}' - ARE YOU EVEN TRYING?!",
    4: "I'M GOING TO KEEP BOTHERING YOU ABOUT '{task}' UNTIL YOU DO IT!!!",
    5: "'{task}' - WHAT IS WRONG WITH YOU?! JUST DO IT ALREADY!!!",
}

def get_nag_message(task_name: str, nag_level: int) -> str:
    """Get the appropriate nag message based on nag level."""
    if nag_level >= 5:
        # Random insults for high nag levels
        insults = [
            f"STILL IGNORING '{task_name.upper()}'?! MY PATIENCE IS WEARING THIN!!!",
            f"'{task_name.upper()}' - DO YOU EVEN CARE ABOUT YOUR RESPONSIBILITIES?!",
            f"I CAN'T BELIEVE YOU HAVEN'T DONE '{task_name.upper()}' YET. PATHETIC!!!",
            f"'{task_name.upper()}' IS MOCKING YOU FROM THE TASK LIST. FINISH IT!!!",
            f"EXCUSE ME?! '{task_name.upper()}' HAS BEEN WAITING FOR AGES!!!",
        ]
        import random
        return random.choice(insults)
    
    template = NAG_MESSAGES.get(nag_level, NAG_MESSAGES[5])
    # Uppercase task name for angry messages (level 3+)
    display_name = task_name.upper() if nag_level >= 3 else task_name
    return template.format(task=display_name)


def check_rate_limit(user_id: int) -> tuple[bool, int]:
    """Check if user has exceeded rate limit. Returns (is_allowed, remaining_requests)."""
    now = get_now()
    hour_ago = now - timedelta(hours=1)
    
    # Clean up old entries
    llm_usage_tracker[user_id] = [
        ts for ts in llm_usage_tracker[user_id] if ts > hour_ago
    ]
    
    current_usage = len(llm_usage_tracker[user_id])
    remaining = LLM_RATE_LIMIT_PER_HOUR - current_usage
    
    if current_usage >= LLM_RATE_LIMIT_PER_HOUR:
        return False, 0
    
    return True, remaining


def record_llm_usage(user_id: int):
    """Record an LLM API call for rate limiting."""
    llm_usage_tracker[user_id].append(get_now())


async def parse_task_with_llm(text: str) -> Optional[dict]:
    """Use Gemini to extract task name and due date from natural language."""
    if not GEMINI_API_KEY:
        return None
    
    try:
        model = genai.GenerativeModel('gemini-2.0-flash-lite')
        
        current_time = get_now().strftime('%Y-%m-%d %H:%M')
        
        prompt = f"""You are a task extraction assistant. Extract the task and deadline from the user's message.

Current date and time: {current_time}

User message: "{text}"

Rules:
- Extract what the user wants to be reminded about
- Convert relative times (tomorrow, in 2 hours, next week) to absolute datetime
- If no specific time given, return understood: false
- For vague times like "tomorrow", assume 9:00 AM
- For "tomorrow afternoon", assume 2:00 PM
- For "tomorrow evening", assume 6:00 PM

Respond ONLY with valid JSON in this exact format, no other text:
{{"task": "task description", "due": "YYYY-MM-DD HH:MM", "understood": true}}

If you cannot extract both a task AND a time, respond:
{{"understood": false}}"""

        response = await model.generate_content_async(prompt)
        response_text = response.text.strip()
        
        # Clean up response - remove markdown code blocks if present
        if response_text.startswith("```"):
            lines = response_text.split("\n")
            response_text = "\n".join(lines[1:-1])
        
        result = json.loads(response_text)
        logger.info(f"LLM parsed: {text} -> {result}")
        return result
        
    except Exception as e:
        logger.error(f"LLM parsing error: {e}")
        return None


def init_database():
    """Initialize the PostgreSQL database with the tasks table."""
    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tasks (
                    id SERIAL PRIMARY KEY,
                    task_name TEXT NOT NULL,
                    due_date TIMESTAMP,
                    status TEXT DEFAULT 'PENDING',
                    nag_level INTEGER DEFAULT 0,
                    chat_id BIGINT,
                    last_nag_time TIMESTAMP
                );
            """)
        logger.info("Database initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")


def add_task(task_name: str, due_date: datetime, chat_id: int) -> int:
    """Add a new task to the database."""
    with conn.cursor() as cursor:
        cursor.execute("""
            INSERT INTO tasks (task_name, due_date, status, nag_level, chat_id, last_nag_time)
            VALUES (%s, %s, 'PENDING', 0, %s, NULL)
            RETURNING id
        """, (task_name, due_date, chat_id))
        
        task_id = cursor.fetchone()[0]
    
    return task_id


def get_pending_tasks(chat_id: int) -> list:
    """Get all pending tasks for a specific chat."""
    with conn.cursor() as cursor:
        cursor.execute("""
            SELECT id, task_name, due_date FROM tasks
            WHERE chat_id = %s AND status = 'PENDING'
            ORDER BY due_date ASC
        """, (chat_id,))
        
        tasks = cursor.fetchall()
        
    return tasks


def get_overdue_pending_tasks() -> list:
    """Get all overdue pending tasks across all chats."""
    now = get_now()
    
    with conn.cursor() as cursor:
        cursor.execute("""
            SELECT id, task_name, due_date, nag_level, chat_id, last_nag_time FROM tasks
            WHERE status = 'PENDING' AND due_date < %s
        """, (now,))
        
        tasks = cursor.fetchall()
        
    return tasks


def update_task_nag(task_id: int):
    """Increment nag level and update last nag time for a task."""
    now = get_now()
    
    with conn.cursor() as cursor:
        cursor.execute("""
            UPDATE tasks
            SET nag_level = nag_level + 1, last_nag_time = %s
            WHERE id = %s
        """, (now, task_id))


def mark_task_done(task_id: int) -> Optional[str]:
    """Mark a task as done and return the task name."""
    with conn.cursor() as cursor:
        # Get task name first
        cursor.execute("SELECT task_name FROM tasks WHERE id = %s", (task_id,))
        result = cursor.fetchone()
        
        if result:
            task_name = result[0]
            cursor.execute("""
                UPDATE tasks SET status = 'DONE' WHERE id = %s
            """, (task_id,))
            return task_name
    
    return None


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle the /start command."""
    ai_note = ""
    if GEMINI_API_KEY:
        ai_note = "\n🧠 *AI Powered:* Just tell me what to remind you about!\n_Example: \"remind me to call mom tomorrow at 3pm\"_\n"
    
    welcome_message = (
        "👋 *Welcome to The Nagger!*\n\n"
        "I'm here to make sure you actually do what you say you will.\n"
        f"{ai_note}\n"
        "*Commands:*\n"
        "• `/add <task> <time>` - Add a new task\n"
        "• `/done` - Mark a task as completed\n"
        "• `/list` - View all pending tasks\n"
        "• `/tutorial` - Step-by-step guide\n"
        "• `/help` - Show this message\n\n"
        "_Fair warning: I get increasingly annoying if you miss deadlines._ 😈"
    )
    
    await update.message.reply_text(welcome_message, parse_mode="Markdown")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle the /help command."""
    await start_command(update, context)


async def tutorial_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle the /tutorial command with a step-by-step guide."""
    ai_section = ""
    if GEMINI_API_KEY:
        ai_section = (
            "━━━━━━━━━━━━━━━━━━━━━\n"
            "*🧠 AI Mode (Easiest!)*\n"
            "━━━━━━━━━━━━━━━━━━━━━\n"
            "Just type naturally! No commands needed:\n\n"
            "💬 _\"remind me to buy milk tomorrow\"_\n"
            "💬 _\"call mom at 5pm\"_\n"
            "💬 _\"submit report in 2 hours\"_\n\n"
            "I'll understand and create the task for you!\n"
            "_(20 AI requests per hour limit)_\n\n"
        )
    
    tutorial_message = (
        "📚 *The Nagger Tutorial*\n\n"
        f"{ai_section}"
        "━━━━━━━━━━━━━━━━━━━━━\n"
        "*Step 1: Add a Task*\n"
        "━━━━━━━━━━━━━━━━━━━━━\n"
        "Use `/add` followed by your task and when it's due:\n\n"
        "✅ `/add Buy milk in 10 minutes`\n"
        "✅ `/add Call mom tomorrow at 5pm`\n"
        "✅ `/add Go to gym in 2 hours`\n\n"
        "━━━━━━━━━━━━━━━━━━━━━\n"
        "*Step 2: View Your Tasks*\n"
        "━━━━━━━━━━━━━━━━━━━━━\n"
        "Type `/list` to see all pending tasks.\n\n"
        "━━━━━━━━━━━━━━━━━━━━━\n"
        "*Step 3: Complete a Task*\n"
        "━━━━━━━━━━━━━━━━━━━━━\n"
        "Type `/done` and tap the button for the task you finished.\n\n"
        "━━━━━━━━━━━━━━━━━━━━━\n"
        "*⚠️ Miss a Deadline?*\n"
        "━━━━━━━━━━━━━━━━━━━━━\n"
        "I'll nag you with increasingly annoying messages! 😈"
    )
    
    await update.message.reply_text(tutorial_message, parse_mode="Markdown")


async def add_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle the /add command to create a new task."""
    if not context.args:
        await update.message.reply_text(
            "❌ Please provide a task and time.\n"
            "_Example: /add Buy groceries in 2 hours_",
            parse_mode="Markdown"
        )
        return
    
    # Join all arguments
    full_text = " ".join(context.args)
    
    # Time keywords to look for - these indicate where the time portion starts
    time_keywords = [
        ' in ', ' at ', ' by ', ' on ', ' tomorrow', ' today', ' tonight',
        ' next ', ' this monday', ' this tuesday', ' this wednesday', 
        ' this thursday', ' this friday', ' this saturday', ' this sunday',
        ' monday', ' tuesday', ' wednesday', ' thursday', ' friday', 
        ' saturday', ' sunday'
    ]
    
    lower_text = " " + full_text.lower()  # Add space prefix for matching
    
    # Find the earliest position of any time keyword
    earliest_pos = -1
    matched_keyword = None
    for keyword in time_keywords:
        pos = lower_text.find(keyword)
        if pos != -1 and (earliest_pos == -1 or pos < earliest_pos):
            earliest_pos = pos
            matched_keyword = keyword
    
    # Extract task name and time portion
    if earliest_pos > 0:
        task_name = full_text[:earliest_pos].strip()
        time_text = full_text[earliest_pos:].strip()
    else:
        # If no keyword found, try the whole thing as time (might be just "2pm" etc)
        task_name = ""
        time_text = full_text
    
    # Try to parse the time portion
    parsed_date = dateparser.parse(
        time_text,
        settings={
            'PREFER_DATES_FROM': 'future',
            'RELATIVE_BASE': get_now()
        }
    )
    
    # If that didn't work, try the full text
    if not parsed_date:
        parsed_date = dateparser.parse(
            full_text,
            settings={
                'PREFER_DATES_FROM': 'future',
                'RELATIVE_BASE': get_now()
            }
        )
    
    if not parsed_date:
        await update.message.reply_text(
            "❌ Couldn't understand the time. Try something like:\n"
            "• `in 10 minutes`\n"
            "• `tomorrow at 5pm`\n"
            "• `next Monday at noon`",
            parse_mode="Markdown"
        )
        return
    
    # If task name is empty, use a default
    if not task_name:
        task_name = "Unnamed task"
    
    # Check if the date is in the past
    if parsed_date <= get_now():
        await update.message.reply_text(
            "❌ That time is in the past. Set a future deadline!",
            parse_mode="Markdown"
        )
        return
    
    # Add the task
    chat_id = update.effective_chat.id
    task_id = add_task(task_name, parsed_date, chat_id)
    
    # Format the date nicely
    formatted_date = parsed_date.strftime("%B %d, %Y at %I:%M %p")
    
    await update.message.reply_text(
        f"✅ Got it! I'll harass you about *\"{task_name}\"* starting *{formatted_date}*.\n\n"
        f"_Don't make me nag you..._ 😏",
        parse_mode="Markdown"
    )
    
    logger.info(f"Task added: '{task_name}' due {parsed_date} for chat {chat_id}")


async def list_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle the /list command to show pending tasks."""
    chat_id = update.effective_chat.id
    tasks = get_pending_tasks(chat_id)
    
    if not tasks:
        await update.message.reply_text(
            "📭 No pending tasks! You're either very productive or very lazy about adding tasks.",
            parse_mode="Markdown"
        )
        return
    
    message = "📋 *Your Pending Tasks:*\n\n"
    
    for task_id, task_name, due_date in tasks:
        # due_date from postgres is already datetime
        if due_date.tzinfo is None:
             due_date = due_date.replace(tzinfo=LOCAL_TZ)
        now = get_now()
        
        if due_date < now:
            status = "⚠️ OVERDUE"
        else:
            time_left = due_date - now
            if time_left.days > 0:
                status = f"⏰ {time_left.days}d left"
            elif time_left.seconds >= 3600:
                hours = time_left.seconds // 3600
                status = f"⏰ {hours}h left"
            else:
                minutes = time_left.seconds // 60
                status = f"⏰ {minutes}m left"
        
        formatted_date = due_date.strftime("%b %d at %I:%M %p")
        message += f"• *{task_name}*\n  Due: {formatted_date} ({status})\n\n"
    
    await update.message.reply_text(message, parse_mode="Markdown")


async def done_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle the /done command to mark tasks as complete."""
    chat_id = update.effective_chat.id
    tasks = get_pending_tasks(chat_id)
    
    if not tasks:
        await update.message.reply_text(
            "📭 No pending tasks to complete!",
            parse_mode="Markdown"
        )
        return
    
    # Create inline keyboard with task buttons
    keyboard = []
    for task_id, task_name, due_date in tasks:
        # Truncate long task names for button
        display_name = task_name[:30] + "..." if len(task_name) > 30 else task_name
        keyboard.append([
            InlineKeyboardButton(f"✓ {display_name}", callback_data=f"done_{task_id}")
        ])
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(
        "🎯 *Which task did you complete?*\n_Tap to mark as done:_",
        reply_markup=reply_markup,
        parse_mode="Markdown"
    )


async def done_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle callback when user clicks a done button."""
    query = update.callback_query
    await query.answer()
    
    # Extract task ID from callback data
    callback_data = query.data
    if not callback_data.startswith("done_"):
        return
    
    task_id = int(callback_data.replace("done_", ""))
    task_name = mark_task_done(task_id)
    
    if task_name:
        # Random congratulations messages
        congrats = [
            f"🎉 Finally. Good job on finishing *\"{task_name}\"*!",
            f"✅ *\"{task_name}\"* is done. Knew you had it in you... eventually.",
            f"🙌 *\"{task_name}\"* completed! One less thing for me to nag about.",
            f"💪 *\"{task_name}\"* is off the list. I'll stop bothering you about it... for now.",
            f"🏆 *\"{task_name}\"* done! See? Was that so hard?",
        ]
        import random
        message = random.choice(congrats)
        
        await query.edit_message_text(message, parse_mode="Markdown")
        logger.info(f"Task completed: '{task_name}' (ID: {task_id})")
    else:
        await query.edit_message_text("❌ Couldn't find that task. Maybe it's already done?")


async def nag_check(context: ContextTypes.DEFAULT_TYPE):
    """Check for overdue tasks and send nag messages."""
    now = get_now()
    logger.info(f"Running nag check at {now.isoformat()}... Found {len(get_overdue_pending_tasks())} overdue tasks.")
    
    overdue_tasks = get_overdue_pending_tasks()
    
    for task_id, task_name, due_date, nag_level, chat_id, last_nag_time in overdue_tasks:
        # Check if enough time has passed since last nag
        if last_nag_time:
            # last_nag_time from postgres is already datetime
            if last_nag_time.tzinfo is None:
                last_nag_time = last_nag_time.replace(tzinfo=LOCAL_TZ)
            time_since_nag = (now - last_nag_time).total_seconds() / 60  # in minutes
            
            if time_since_nag < NAG_INTERVAL_MINUTES:
                logger.debug(f"Skipping nag for task {task_id}, only {time_since_nag:.1f} min since last nag")
                continue
        
        # Get appropriate nag message
        message = get_nag_message(task_name, nag_level)
        
        try:
            await context.bot.send_message(chat_id=chat_id, text=message)
            update_task_nag(task_id)
            logger.info(f"Nagged about task '{task_name}' (level {nag_level}) to chat {chat_id}")
        except Exception as e:
            logger.error(f"Failed to send nag for task {task_id}: {e}")


async def natural_language_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle natural language messages and try to extract tasks using LLM."""
    if not GEMINI_API_KEY:
        # LLM not configured, ignore non-command messages
        return
    
    text = update.message.text
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id
    
    # Skip very short messages or ones that look like commands
    if len(text) < 5 or text.startswith('/'):
        return
    
    # Check rate limit
    is_allowed, remaining = check_rate_limit(user_id)
    if not is_allowed:
        await update.message.reply_text(
            "⚠️ *Rate limit reached!*\n"
            "You've used all 20 AI requests this hour.\n"
            "Try again later, or use `/add` command instead.",
            parse_mode="Markdown"
        )
        return
    
    # Record usage and call LLM
    record_llm_usage(user_id)
    result = await parse_task_with_llm(text)
    
    if not result or not result.get("understood"):
        # LLM didn't understand - silently ignore to avoid annoying users
        # with every message they send
        return
    
    try:
        task_name = result.get("task", "").strip()
        due_str = result.get("due", "")
        
        if not task_name or not due_str:
            return
        
        # Parse the due date and make it timezone-aware
        parsed_date = datetime.strptime(due_str, "%Y-%m-%d %H:%M").replace(tzinfo=LOCAL_TZ)
        
        # Check if in past
        if parsed_date <= get_now():
            await update.message.reply_text(
                "🤔 That time seems to be in the past. Try a future deadline!",
                parse_mode="Markdown"
            )
            return
        
        # Add the task
        task_id = add_task(task_name, parsed_date, chat_id)
        formatted_date = parsed_date.strftime("%B %d, %Y at %I:%M %p")
        
        await update.message.reply_text(
            f"🧠 *Got it!* I understood your message.\n\n"
            f"📝 Task: *\"{task_name}\"*\n"
            f"⏰ Due: *{formatted_date}*\n\n"
            f"_I'll start nagging you after the deadline!_ 😈\n\n"
            f"_{remaining - 1} AI requests remaining this hour_",
            parse_mode="Markdown"
        )
        
        logger.info(f"Task added via LLM: '{task_name}' due {parsed_date} for chat {chat_id}")
        
    except Exception as e:
        logger.error(f"Error processing LLM result: {e}")


async def main():
    """Main entry point for the bot."""
    if not BOT_TOKEN:
        logger.error("TELEGRAM_BOT_TOKEN not found in environment variables!")
        logger.error("Please create a .env file with your bot token.")
        logger.error("Example: TELEGRAM_BOT_TOKEN=your_token_here")
        return
    
    # Initialize database
    init_database()
    
    # Create the Application
    application = Application.builder().token(BOT_TOKEN).build()
    
    # Add command handlers
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("tutorial", tutorial_command))
    application.add_handler(CommandHandler("add", add_command))
    application.add_handler(CommandHandler("list", list_command))
    application.add_handler(CommandHandler("done", done_command))
    
    # Add callback handler for done buttons
    application.add_handler(CallbackQueryHandler(done_callback, pattern="^done_"))
    
    # Add natural language handler (processes any text message with LLM)
    if GEMINI_API_KEY:
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, natural_language_handler))
        logger.info("Gemini LLM enabled - natural language processing active")
    else:
        logger.warning("GEMINI_API_KEY not set - natural language processing disabled")
    
    # Set up the job queue for nagging (runs every 60 seconds)
    job_queue = application.job_queue
    job_queue.run_repeating(nag_check, interval=10, first=10)
    
    logger.info("The Nagger bot is starting...")
    logger.info("Press Ctrl+C to stop.")
    
    # Initialize and run the bot using the async context manager
    async with application:
        await application.start()
        await application.updater.start_polling(allowed_updates=Update.ALL_TYPES)
        
        # Keep running until interrupted
        try:
            while True:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass
        finally:
            await application.updater.stop()
            await application.stop()


if __name__ == "__main__":
    asyncio.run(main())
