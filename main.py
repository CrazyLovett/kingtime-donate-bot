import asyncio
import os
import random
import re
import signal
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

import aiosqlite
import yaml
from aiohttp import web

from aiogram import Bot, Dispatcher, F
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import (
    Message,
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup
)

# ================== PATHS ==================
BASE_DIR = Path(__file__).parent
CONFIG_PATH = BASE_DIR / "config.yaml"
DB_PATH = BASE_DIR / "db.sqlite3"

NICK_RE = re.compile(r"^[A-Za-z0-9_]{3,16}$")

# ================== ENV ==================
def env(name: str, default: str = "") -> str:
    return os.getenv(name, default)

# ================== LOAD CONFIG ==================
def load_cfg() -> Dict[str, Any]:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["bot"]["token"] = env("BOT_TOKEN")
    cfg["api"]["token"] = env("API_TOKEN")
    cfg["payment"]["card"] = env("CARD_NUMBER")

    if not cfg["bot"]["token"]:
        raise RuntimeError("BOT_TOKEN not set")
    if not cfg["api"]["token"]:
        raise RuntimeError("API_TOKEN not set")
    if not cfg["payment"]["card"]:
        raise RuntimeError("CARD_NUMBER not set")

    return cfg

# ================== STATES ==================
class BuyFlow(StatesGroup):
    nick = State()
    receipt = State()

# ================== HELPERS ==================
def gen_code(prefix: str, length: int) -> str:
    chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    return prefix + "-" + "".join(random.choice(chars) for _ in range(length))

def pretty_card(card: str) -> str:
    return " ".join(card[i:i+4] for i in range(0, len(card), 4))

# ================== DATABASE ==================
async def db_init():
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
        CREATE TABLE IF NOT EXISTS orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created TEXT,
            tg_id INTEGER,
            nick TEXT,
            product TEXT,
            price INTEGER,
            code TEXT,
            status TEXT
        )
        """)
        await db.commit()

async def db_add(order: dict):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO orders VALUES (NULL,?,?,?,?,?,?)",
            (
                order["created"],
                order["tg_id"],
                order["nick"],
                order["product"],
                order["price"],
                order["code"],
                order["status"]
            )
        )
        await db.commit()

# ================== MAIN ==================
async def main():
    cfg = load_cfg()
    await db_init()

    bot = Bot(
        cfg["bot"]["token"],
        default=DefaultBotProperties(parse_mode=ParseMode.HTML)
    )
    dp = Dispatcher()

    # --------- BOT ---------
    @dp.message(CommandStart())
    async def start(m: Message, state: FSMContext):
        await state.clear()
        kb = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="💎 Купить донат", callback_data="buy")]
        ])
        await m.answer(
            "🛒 <b>KingTime | Донат магазин</b>\n\nВыбери действие:",
            reply_markup=kb
        )

    @dp.callback_query(F.data == "buy")
    async def buy(cq: CallbackQuery, state: FSMContext):
        await state.set_state(BuyFlow.nick)
        await cq.message.edit_text("✍️ Введи ник игрока:")
        await cq.answer()

    @dp.message(BuyFlow.nick)
    async def get_nick(m: Message, state: FSMContext):
        nick = m.text.strip()
        if not NICK_RE.match(nick):
            await m.answer("❌ Ник неверный, попробуй ещё раз")
            return

        code = gen_code(cfg["payment"]["comment_prefix"], cfg["payment"]["code_length"])

        order = {
            "created": datetime.now().isoformat(timespec="seconds"),
            "tg_id": m.from_user.id,
            "nick": nick,
            "product": "DONATE",
            "price": 100,
            "code": code,
            "status": "WAITING"
        }

        await db_add(order)
        await state.clear()

        await m.answer(
            f"💳 <b>Оплата</b>\n\n"
            f"💰 Сумма: <b>{order['price']} ₽</b>\n"
            f"🏷 Код (В НАЧАЛЕ комментария): <code>{code}</code>\n\n"
            f"👤 Получатель: {cfg['payment']['fio']}\n"
            f"💳 Карта: <code>{pretty_card(cfg['payment']['card'])}</code>\n\n"
            f"После оплаты отправь чек администрации."
        )

    # --------- HTTP (для Render) ---------
    async def index(_):
        return web.Response(text="OK")

    app = web.Application()
    app.router.add_get("/", index)

    port = int(env("PORT", "10000"))
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()

    stop_event = asyncio.Event()

    def stop(*_):
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig,sig)
        except:
            pass

    polling = asyncio.create_task(dp.start_polling(bot))
    await stop_event.wait()

    polling.cancel()
    await bot.session.close()
    await runner.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
