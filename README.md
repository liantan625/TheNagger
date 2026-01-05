# 🔔 The Nagger

A Telegram bot that accepts tasks with deadlines and sends **increasingly annoying messages** if the task is not marked as done after the deadline.

## Features

- **Natural language dates** - Add tasks using human-friendly time descriptions like "in 10 minutes" or "tomorrow at 5pm"
- **Smart nagging** - Messages escalate from polite reminders to ALL CAPS HARASSMENT
- **Inline buttons** - Easy task completion with one tap
- **Persistent storage** - Tasks survive bot restarts using SQLite

## Setup

### 1. Create a Telegram Bot

1. Open Telegram and search for [@BotFather](https://t.me/BotFather)
2. Send `/newbot` and follow the prompts
3. Copy the API token you receive

### 2. Configure Environment

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your bot token
TELEGRAM_BOT_TOKEN=your_token_here
```

### 3. Install Dependencies

```bash
# Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 4. Run the Bot

```bash
python main.py
```

## Commands

| Command | Description | Example |
|---------|-------------|---------|
| `/start` | Welcome message and help | `/start` |
| `/add <task> <time>` | Add a new task with deadline | `/add Buy groceries in 2 hours` |
| `/list` | View all pending tasks | `/list` |
| `/done` | Mark a task as completed | `/done` |
| `/help` | Show help message | `/help` |

## Time Format Examples

- `in 10 minutes`
- `in 2 hours`
- `tomorrow at 5pm`
- `next Monday at noon`
- `this Friday evening`

## How Nagging Works

1. **Nag Level 0**: _"Hey, [task] is due. Maybe get on it?"_
2. **Nag Level 1**: _"You said you'd do [task]. Do it."_
3. **Nag Level 2**: _"Seriously? [task] is STILL not done?"_
4. **Nag Level 3+**: ALL CAPS MESSAGES AND MILD INSULTS 😈

The bot checks every minute but only sends messages every 5 minutes to avoid spamming.

## Tech Stack

- **python-telegram-bot** - Async Telegram Bot API wrapper
- **APScheduler** - Background task scheduling
- **dateparser** - Natural language date parsing
- **SQLite3** - Lightweight persistent storage

## License

MIT
