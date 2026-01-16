import asyncio
import json
import os
import random
import re
import signal
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiosqlite
import yaml
from aiohttp import web

from aiogram import Bot, Dispatcher, F
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart, Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)

CONFIG_PATH = Path("config.yaml")
DB_PATH = Path("db.sqlite3")
NICK_RE = re.compile(r"^[A-Za-z0-9_]{3,16}$")


class BuyFlow(StatesGroup):
    entering_nick = State()
    waiting_receipt = State()


@dataclass
class Product:
    key: str
    title: str
    price_rub: int
    commands: List[str]
    announce: str


def load_cfg() -> Dict[str, Any]:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def gen_code(prefix: str, length: int) -> str:
    chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    return prefix + "-" + "".join(random.choice(chars) for _ in range(length))


def is_admin(cfg: Dict[str, Any], uid: int) -> bool:
    return uid in cfg.get("admins", [])


def get_product(cfg: Dict[str, Any], key: str) -> Product:
    d = cfg["products"][key]
    return Product(
        key=key,
        title=str(d["title"]),
        price_rub=int(d["price_rub"]),
        commands=list(d.get("commands", [])),
        announce=str(d.get("announce", d["title"]))
    )


def list_products(cfg: Dict[str, Any]) -> List[Product]:
    items = [get_product(cfg, k) for k in cfg["products"].keys()]
    items.sort(key=lambda x: (x.price_rub, x.title.lower()))
    return items


def kb_shop(products: List[Product]) -> InlineKeyboardMarkup:
    rows = []
    for p in products:
        rows.append([InlineKeyboardButton(
            text=f"{p.title} — {p.price_rub}₽",
            callback_data=f"buy:{p.key}"
        )])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def kb_after_pay(order_id: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton("✅ Я оплатил (отправить чек)", callback_data=f"paid:{order_id}")],
        [InlineKeyboardButton("⬅️ Назад в магазин", callback_data="shop")]
    ])


# ---------------- DB ----------------

async def db_init():
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
        CREATE TABLE IF NOT EXISTS orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT,
            tg_user_id INTEGER,
            tg_username TEXT,
            nick TEXT,
            product_key TEXT,
            product_title TEXT,
            amount_rub INTEGER,
            code TEXT,
            status TEXT,
            receipt_file_id TEXT
        )
        """)
        await db.commit()


async def db_create_order(tg_id: int, username: Optional[str], nick: str, product: Product, code: str) -> int:
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute("""
            INSERT INTO orders
            (created_at,tg_user_id,tg_username,nick,product_key,product_title,amount_rub,code,status,receipt_file_id)
            VALUES(?,?,?,?,?,?,?,?,?,?)
        """, (
            datetime.now().isoformat(timespec="seconds"),
            tg_id,
            username,
            nick,
            product.key,
            product.title,
            product.price_rub,
            code,
            "waiting_receipt",
            None
        ))
        await db.commit()
        return cur.lastrowid


async def db_get(order_id: int) -> Optional[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute("SELECT * FROM orders WHERE id=?", (order_id,))
        r = await cur.fetchone()
        return dict(r) if r else None


# ---------------- HTTP (Healthcheck) ----------------

async def index(request: web.Request) -> web.Response:
    return web.Response(text="OK", content_type="text/plain")


async def api_pull(request: web.Request) -> web.Response:
    # Заглушка, чтобы агент Bothost видел 200 и JSON
    return web.json_response({"ok": True, "orders": []})


async def api_ack(request: web.Request) -> web.Response:
    return web.json_response({"ok": True})


async def start_http_server(cfg: Dict[str, Any]) -> tuple[web.AppRunner, web.TCPSite]:
    app = web.Application()
    app.router.add_get("/", index)
    app.router.add_get("/api/pull", api_pull)
    app.router.add_post("/api/ack", api_ack)

    port = int(os.getenv("PORT", str(cfg.get("api", {}).get("port", 8080))))
    host = "0.0.0.0"

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()
    print(f"HTTP server: {host}:{port}")
    return runner, site


# ---------------- MAIN ----------------

async def run_bot(cfg: Dict[str, Any], stop_event: asyncio.Event):
    bot = Bot(
        cfg["bot"]["token"],
        default=DefaultBotProperties(parse_mode=ParseMode.HTML)
    )
    dp = Dispatcher()
    products = list_products(cfg)

    @dp.message(CommandStart())
    async def start_cmd(m: Message, state: FSMContext):
        await state.clear()
        await m.answer("🛒 <b>Магазин доната KingTime</b>", reply_markup=kb_shop(products))

    @dp.message(Command("donate"))
    async def donate_cmd(m: Message, state: FSMContext):
        await state.clear()
        await m.answer("🛒 <b>Магазин доната KingTime</b>", reply_markup=kb_shop(products))

    @dp.callback_query(F.data == "shop")
    async def back_shop(cq: CallbackQuery, state: FSMContext):
        await state.clear()
        await cq.message.edit_text("🛒 <b>Магазин доната KingTime</b>", reply_markup=kb_shop(products))
        await cq.answer()

    @dp.callback_query(F.data.startswith("buy:"))
    async def buy(cq: CallbackQuery, state: FSMContext):
        key = cq.data.split(":", 1)[1]
        if key not in cfg["products"]:
            await cq.answer("Товар не найден", show_alert=True)
            return
        await state.set_state(BuyFlow.entering_nick)
        await state.update_data(product_key=key)
        p = get_product(cfg, key)
        await cq.message.edit_text(f"📦 <b>{p.title}</b>\n💰 {p.price_rub} ₽\n\n✍️ Введи ник:")
        await cq.answer()

    @dp.message(BuyFlow.entering_nick)
    async def got_nick(m: Message, state: FSMContext):
        nick = (m.text or "").strip()
        if not NICK_RE.match(nick):
            await m.answer("❌ Неверный ник. Пример: <code>Steve_123</code>\nПопробуй снова:")
            return

        data = await state.get_data()
        key = data["product_key"]
        p = get_product(cfg, key)

        code = gen_code(cfg["payment"]["comment_prefix"], int(cfg["payment"]["code_length"]))
        order_id = await db_create_order(m.from_user.id, m.from_user.username, nick, p, code)

        await state.clear()
        await m.answer(
            f"💳 Сумма: <b>{p.price_rub} ₽</b>\n🏷 Код: <code>{code}</code>\n\n"
            "Дальше будет выдача после подтверждения админом ✅",
            reply_markup=kb_after_pay(order_id)
        )

    async def polling_task():
        # start_polling будет жить пока не отменим
        await dp.start_polling(bot)

    task = asyncio.create_task(polling_task())
    try:
        # ждём сигнал остановки
        await stop_event.wait()
    finally:
        # корректно закрываем
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception:
            pass
        await bot.session.close()


async def main():
    cfg = load_cfg()
    await db_init()

    stop_event = asyncio.Event()

    # Ловим сигналы stop от хостинга
    loop = asyncio.get_running_loop()

    def _stop(*_):
        print("Stopping...")
        stop_event.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, _stop)
        except NotImplementedError:
            # на некоторых платформах может не поддерживаться
            pass

    runner, _site = await start_http_server(cfg)

    bot_task = asyncio.create_task(run_bot(cfg, stop_event))

    try:
        await bot_task
    finally:
        # выключаем HTTP
        try:
            await runner.cleanup()
        except Exception:
            pass


if __name__ == "__main__":
    asyncio.run(main())
