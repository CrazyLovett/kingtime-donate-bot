import asyncio
import logging
import os
import random
import re
import signal
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import aiosqlite
import yaml
from aiohttp import web

from aiogram import Bot, Dispatcher, F
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart, Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message

# ===================== LOGGING =====================
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("kingtime-bot")

# ===================== PATHS =====================
BASE_DIR = Path(__file__).parent
CONFIG_PATH = BASE_DIR / "config.yaml"
DB_PATH = BASE_DIR / "db.sqlite3"

NICK_RE = re.compile(r"^[A-Za-z0-9_]{3,16}$")


def env(name: str, default: str = "") -> str:
    return os.getenv(name, default)


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


# ===================== LOAD CONFIG =====================
def load_cfg() -> Dict[str, Any]:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    # делаем структуру “неубиваемой”
    cfg.setdefault("app", {})
    cfg.setdefault("bot", {})
    cfg.setdefault("api", {})
    cfg.setdefault("payment", {})
    cfg.setdefault("products", {})
    cfg.setdefault("admins", [])

    # secrets only from ENV
    cfg["bot"]["token"] = env("BOT_TOKEN")
    cfg["api"]["token"] = env("API_TOKEN")
    cfg["payment"]["card"] = env("CARD_NUMBER")

    cfg["app"]["public_url"] = env("PUBLIC_URL", cfg["app"].get("public_url", ""))
    cfg["app"]["bot_link"] = cfg["app"].get("bot_link", "https://t.me/KingTimeDonateBot")

    # проверки
    if not cfg["bot"]["token"]:
        raise RuntimeError("ENV BOT_TOKEN not set")
    if not cfg["api"]["token"]:
        raise RuntimeError("ENV API_TOKEN not set")
    if not cfg["payment"]["card"]:
        raise RuntimeError("ENV CARD_NUMBER not set")
    if not isinstance(cfg["products"], dict) or len(cfg["products"]) == 0:
        raise RuntimeError("config.yaml: products is empty or not a dict")

    return cfg


# ===================== STATES =====================
class BuyFlow(StatesGroup):
    entering_nick = State()
    waiting_receipt = State()


# ===================== HELPERS =====================
def gen_code(prefix: str, length: int) -> str:
    chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    return prefix + "-" + "".join(random.choice(chars) for _ in range(length))


def pretty_card(raw: str) -> str:
    raw = (raw or "").strip().replace(" ", "")
    return " ".join(raw[i:i + 4] for i in range(0, len(raw), 4))


def payment_text(cfg: Dict[str, Any], nick: str, title: str, price: int, code: str) -> str:
    p = cfg["payment"]
    bank = p.get("bank", "")
    return (
        "💳 <b>Как правильно отправить деньги</b>\n\n"
        "❗ <b>ВАЖНО:</b> в комментарии к переводу код должен быть <b>ПЕРВЫМ</b>\n"
        f"🏷 Код: <code>{code}</code>\n\n"
        f"👤 Ник: <code>{nick}</code>\n"
        f"📦 Товар: <b>{title}</b>\n"
        f"💰 Сумма: <b>{price} ₽</b>\n\n"
        "Перевод на карту:\n"
        f"• <b>Получатель:</b> {p.get('fio','')}\n"
        + (f"• <b>Банк:</b> {bank}\n" if bank else "")
        + f"• <b>Номер карты:</b> <code>{pretty_card(p.get('card',''))}</code>\n\n"
        "✅ После оплаты нажми <b>«Я оплатил»</b> и пришли <b>чек/скрин</b>."
    )


# ===================== KEYBOARDS =====================
def kb_products(cfg: Dict[str, Any]) -> InlineKeyboardMarkup:
    items = list(cfg.get("products", {}).items())
    items.sort(key=lambda kv: (int(kv[1].get("price", 0)), str(kv[1].get("title", "")).lower()))

    rows = []
    for key, p in items:
        rows.append([InlineKeyboardButton(
            text=f"{p.get('title','')} — {int(p.get('price',0))}₽",
            callback_data=f"buy:{key}"
        )])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def kb_after_pay(order_id: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="✅ Я оплатил (отправить чек)", callback_data=f"paid:{order_id}")],
        [InlineKeyboardButton(text="⬅️ В магазин", callback_data="shop")],
    ])


# ===================== DB =====================
CREATE_SQL = """
CREATE TABLE IF NOT EXISTS orders (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at TEXT NOT NULL,
  tg_user_id INTEGER NOT NULL,
  tg_username TEXT,
  nick TEXT NOT NULL,
  product_key TEXT NOT NULL,
  product_title TEXT NOT NULL,
  amount_rub INTEGER NOT NULL,
  code TEXT NOT NULL,
  status TEXT NOT NULL,
  receipt_file_id TEXT
)
"""


async def db_init():
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(CREATE_SQL)
        await db.commit()


async def db_create_order(
    tg_user_id: int,
    tg_username: Optional[str],
    nick: str,
    product_key: str,
    product_title: str,
    amount_rub: int,
    code: str,
) -> int:
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute(
            """
            INSERT INTO orders
            (created_at,tg_user_id,tg_username,nick,product_key,product_title,amount_rub,code,status)
            VALUES(?,?,?,?,?,?,?,?,?)
            """,
            (now_iso(), tg_user_id, tg_username, nick, product_key, product_title, amount_rub, code, "waiting_receipt"),
        )
        await db.commit()
        return int(cur.lastrowid)


async def db_get(order_id: int) -> Optional[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute("SELECT * FROM orders WHERE id=?", (order_id,))
        row = await cur.fetchone()
        return dict(row) if row else None


# ===================== HTTP =====================
async def http_index(_req: web.Request) -> web.Response:
    return web.Response(text="OK", content_type="text/plain")


# ===================== MAIN =====================
async def main():
    cfg = load_cfg()
    await db_init()

    bot = Bot(cfg["bot"]["token"], default=DefaultBotProperties(parse_mode=ParseMode.HTML))
    dp = Dispatcher(storage=MemoryStorage())

    # Глобальный перехват ошибок — чтобы в логах Render был полный traceback
    @dp.error()
    async def on_error(event, exception: Exception):
        log.error("UNHANDLED ERROR: %s", repr(exception))
        log.error(traceback.format_exc())
        return True

    # HTTP server for Render
    http_app = web.Application()
    http_app.router.add_get("/", http_index)

    port = int(env("PORT", "10000"))
    runner = web.AppRunner(http_app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()
    log.info("HTTP started on 0.0.0.0:%s", port)

    # ---------------- BOT ----------------
    @dp.message(CommandStart())
    async def start_cmd(m: Message, state: FSMContext):
        await state.clear()
        await m.answer("🛒 <b>KingTime | Донат магазин</b>\nВыбери товар:", reply_markup=kb_products(cfg))

    @dp.message(Command("shop"))
    async def shop_cmd(m: Message, state: FSMContext):
        await state.clear()
        await m.answer("🛒 Выбери товар:", reply_markup=kb_products(cfg))

    @dp.callback_query(F.data == "shop")
    async def shop_cb(cq: CallbackQuery, state: FSMContext):
        await state.clear()
        await cq.message.edit_text("🛒 Выбери товар:", reply_markup=kb_products(cfg))
        await cq.answer()

    @dp.callback_query(F.data.startswith("buy:"))
    async def buy_pick(cq: CallbackQuery, state: FSMContext):
        key = cq.data.split(":", 1)[1]
        p = cfg["products"].get(key)
        if not p:
            await cq.answer("Товар не найден", show_alert=True)
            return

        await state.set_state(BuyFlow.entering_nick)
        await state.update_data(product_key=key)
        await cq.message.edit_text(f"✍️ Введи ник игрока для <b>{p.get('title','')}</b>:")
        await cq.answer()

    @dp.message(BuyFlow.entering_nick)
    async def got_nick(m: Message, state: FSMContext):
        nick = (m.text or "").strip()
        if not NICK_RE.match(nick):
            await m.answer("❌ Ник неверный. Пример: <code>Steve_123</code>\nПопробуй снова:")
            return

        data = await state.get_data()
        key = data.get("product_key")
        p = cfg["products"].get(key)
        if not p:
            await state.clear()
            await m.answer("❌ Товар не найден. Нажми /start")
            return

        code = gen_code(cfg["payment"].get("comment_prefix", "KT"), int(cfg["payment"].get("code_length", 8)))
        order_id = await db_create_order(
            tg_user_id=m.from_user.id,
            tg_username=m.from_user.username,
            nick=nick,
            product_key=key,
            product_title=str(p.get("title", key)),
            amount_rub=int(p.get("price", 0)),
            code=code,
        )
        await state.clear()

        await m.answer(
            payment_text(cfg, nick, str(p.get("title", key)), int(p.get("price", 0)), code),
            reply_markup=kb_after_pay(order_id),
        )

    # graceful shutdown
    stop_event = asyncio.Event()

    def _stop():
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _stop)
        except NotImplementedError:
            pass

    polling_task = asyncio.create_task(dp.start_polling(bot))
    await stop_event.wait()

    polling_task.cancel()
    try:
        await polling_task
    except Exception:
        pass

    await bot.session.close()
    await runner.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
