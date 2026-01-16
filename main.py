import asyncio
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
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message

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
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    return f"{prefix}-" + "".join(random.choice(alphabet) for _ in range(length))


def pretty_card(raw: str) -> str:
    return " ".join(raw[i:i + 4] for i in range(0, len(raw), 4))


def is_admin(cfg: Dict[str, Any], uid: int) -> bool:
    return uid in set(cfg.get("admins", []))


def get_product(cfg: Dict[str, Any], key: str) -> Product:
    d = cfg["products"][key]
    return Product(
        key=key,
        title=str(d["title"]),
        price_rub=int(d["price_rub"]),
        commands=list(d.get("commands", [])),
        announce=str(d.get("announce", d["title"])),
    )


def list_products(cfg: Dict[str, Any]) -> List[Product]:
    items = [get_product(cfg, k) for k in cfg["products"].keys()]
    items.sort(key=lambda p: (p.price_rub, p.title.lower()))
    return items


def payment_text(cfg: Dict[str, Any], amount: int, code: str) -> str:
    p = cfg["payment"]
    bank = p.get("bank", "")
    return (
        "💳 <b>Как правильно отправить деньги</b>\n\n"
        "❗ <b>ВАЖНО:</b> в комментарии к переводу <b>ОБЯЗАТЕЛЬНО</b> укажи код <b>В НАЧАЛЕ</b>\n"
        f"🏷 <b>Код:</b> <code>{code}</code>\n\n"
        "✅ <b>Пошагово:</b>\n"
        f"1) Переведи <b>ТОЧНУЮ сумму</b>: <b>{amount} ₽</b>\n"
        "2) Перевод на карту:\n"
        f"   • <b>Получатель:</b> {p['fio']}\n"
        + (f"   • <b>Банк:</b> {bank}\n" if bank else "")
        + f"   • <b>Номер карты:</b> <code>{pretty_card(p['card'])}</code>\n"
        "3) Комментарий (код должен быть ПЕРВЫМ, потом можно текст):\n"
        f"   Пример: <code>{code} донат</code>\n"
        "4) Нажми кнопку <b>«Я оплатил»</b> и отправь <b>чек/скрин</b>\n\n"
        "⚠️ Нет кода в начале / неверная сумма — заявка будет отклонена."
    )


def kb_shop(products: List[Product]) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text=f"{p.title} — {p.price_rub}₽", callback_data=f"buy:{p.key}")]
            for p in products
        ]
    )


def kb_after_pay(order_id: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="✅ Я оплатил (отправить чек)", callback_data=f"paid:{order_id}")],
        [InlineKeyboardButton(text="⬅️ Назад в магазин", callback_data="shop")],
    ])


def kb_admin(order_id: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(text="✅ Подтвердить", callback_data=f"adm_ok:{order_id}"),
        InlineKeyboardButton(text="❌ Отказать", callback_data=f"adm_no:{order_id}"),
    ]])


def kb_reasons(order_id: int) -> InlineKeyboardMarkup:
    reasons = [
        ("💸 Неверная сумма", "sum"),
        ("🏷 Нет кода в начале", "code"),
        ("🧾 Чек не читается/нет чека", "receipt"),
        ("🔁 Уже выдано ранее", "dup"),
        ("🚫 Подозрение на фейк", "fake"),
        ("✍️ Другая причина", "other"),
        ("⬅️ Назад", "back"),
    ]
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text=t, callback_data=f"adm_reason:{order_id}:{tag}")]
            for (t, tag) in reasons
        ]
    )


async def db_init():
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
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
          status TEXT NOT NULL,            -- waiting_receipt / pending_review / approved / issued / rejected
          receipt_file_id TEXT,
          admin_id INTEGER,
          reject_reason TEXT
        )
        """)
        await db.commit()


async def db_create_order(tg_user_id: int, tg_username: Optional[str], nick: str, product: Product, code: str) -> int:
    now = datetime.now().isoformat(timespec="seconds")
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute("""
        INSERT INTO orders(created_at,tg_user_id,tg_username,nick,product_key,product_title,amount_rub,code,status)
        VALUES(?,?,?,?,?,?,?,?,?)
        """, (now, tg_user_id, tg_username, nick, product.key, product.title, product.price_rub, code, "waiting_receipt"))
        await db.commit()
        return cur.lastrowid


async def db_get(order_id: int) -> Optional[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute("SELECT * FROM orders WHERE id=?", (order_id,))
        row = await cur.fetchone()
        return dict(row) if row else None


async def db_set_receipt(order_id: int, file_id: str):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("UPDATE orders SET receipt_file_id=?, status=? WHERE id=?",
                         (file_id, "pending_review", order_id))
        await db.commit()


async def db_set_status(order_id: int, status: str, admin_id: Optional[int] = None, reason: Optional[str] = None):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
        UPDATE orders
        SET status=?,
            admin_id=COALESCE(?, admin_id),
            reject_reason=COALESCE(?, reject_reason)
        WHERE id=?
        """, (status, admin_id, reason, order_id))
        await db.commit()


def admin_card(order: dict) -> str:
    return (
        "💰 <b>Новая заявка</b>\n\n"
        f"🆔 <b>Заявка:</b> #{order['id']}\n"
        f"👤 <b>Игрок:</b> <code>{order['nick']}</code>\n"
        f"📦 <b>Товар:</b> {order['product_title']}\n"
        f"💵 <b>Сумма:</b> {order['amount_rub']} ₽\n"
        f"🏷 <b>Код:</b> <code>{order['code']}</code>\n"
    )


# OPTIONAL: healthcheck server for host panels (even without public domain)
async def index(_request: web.Request) -> web.Response:
    return web.Response(text="OK", content_type="text/plain")


async def start_local_http():
    # Some platforms require a listening port; harmless otherwise.
    port = int(os.getenv("PORT", "8080"))
    app = web.Application()
    app.router.add_get("/", index)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()
    return runner


async def main():
    cfg = load_cfg()
    await db_init()

    stop_event = asyncio.Event()

    loop = asyncio.get_running_loop()
    def _stop(*_):
        stop_event.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, _stop)
        except NotImplementedError:
            pass

    # start minimal http (optional)
    http_runner = await start_local_http()

    bot = Bot(
        cfg["bot"]["token"],
        default=DefaultBotProperties(parse_mode=ParseMode.HTML)
    )
    dp = Dispatcher()
    products = list_products(cfg)

    @dp.message(CommandStart())
    async def start_cmd(m: Message, state: FSMContext):
        await state.clear()
        await m.answer("🛒 <b>Магазин доната kingtime</b>\nВыбери товар:", reply_markup=kb_shop(products))

    @dp.message(Command("donate"))
    async def donate_cmd(m: Message, state: FSMContext):
        await state.clear()
        await m.answer("🛒 Магазин:", reply_markup=kb_shop(products))

    @dp.callback_query(F.data == "shop")
    async def back_shop(cq: CallbackQuery, state: FSMContext):
        await state.clear()
        await cq.message.edit_text("🛒 Магазин:", reply_markup=kb_shop(products))
        await cq.answer()

    @dp.callback_query(F.data.startswith("buy:"))
    async def choose_product(cq: CallbackQuery, state: FSMContext):
        key = cq.data.split(":", 1)[1]
        if key not in cfg["products"]:
            await cq.answer("Товар не найден", show_alert=True)
            return
        await state.set_state(BuyFlow.entering_nick)
        await state.update_data(product_key=key)
        p = get_product(cfg, key)
        await cq.message.edit_text(
            f"📦 <b>{p.title}</b>\n💵 Цена: <b>{p.price_rub} ₽</b>\n\n"
            "✍️ Отправь <b>ник</b> игрока (A-Z 0-9 _):"
        )
        await cq.answer()

    @dp.message(BuyFlow.entering_nick)
    async def got_nick(m: Message, state: FSMContext):
        nick = (m.text or "").strip()
        if not NICK_RE.match(nick):
            await m.answer("❌ Ник неверный. Пример: <code>Steve_123</code>\nПопробуй ещё раз:")
            return

        data = await state.get_data()
        key = data["product_key"]
        p = get_product(cfg, key)

        code = gen_code(cfg["payment"]["comment_prefix"], int(cfg["payment"]["code_length"]))
        order_id = await db_create_order(m.from_user.id, m.from_user.username, nick, p, code)

        await state.clear()
        await m.answer(payment_text(cfg, p.price_rub, code), reply_markup=kb_after_pay(order_id))

    @dp.callback_query(F.data.startswith("paid:"))
    async def paid_btn(cq: CallbackQuery, state: FSMContext):
        order_id = int(cq.data.split(":", 1)[1])
        order = await db_get(order_id)
        if not order or order["tg_user_id"] != cq.from_user.id:
            await cq.answer("Заявка не найдена", show_alert=True)
            return

        await state.set_state(BuyFlow.waiting_receipt)
        await state.update_data(order_id=order_id)
        await cq.message.edit_text(
            f"🧾 Отправь <b>чек/скрин</b> одним сообщением.\n\n"
            f"Заявка: <b>#{order_id}</b>\nКод: <code>{order['code']}</code>"
        )
        await cq.answer()

    @dp.message(BuyFlow.waiting_receipt, F.photo | F.document)
    async def receipt(m: Message, state: FSMContext):
        data = await state.get_data()
        order_id = int(data["order_id"])
        order = await db_get(order_id)
        if not order or order["tg_user_id"] != m.from_user.id:
            await m.answer("Заявка не найдена.")
            return

        file_id = None
        if m.photo:
            file_id = m.photo[-1].file_id
        elif m.document:
            file_id = m.document.file_id

        if not file_id:
            await m.answer("Не получилось получить файл. Пришли фото/документ ещё раз.")
            return

        await db_set_receipt(order_id, file_id)
        await state.clear()

        await m.answer("✅ Принято! Заявка отправлена на проверку админу.")

        for admin_id in cfg["admins"]:
            try:
                await bot.send_message(admin_id, admin_card(await db_get(order_id)), reply_markup=kb_admin(order_id))
                await bot.send_photo(admin_id, file_id, caption=f"📎 Чек к заявке #{order_id}")
            except Exception:
                pass

    @dp.callback_query(F.data.startswith("adm_ok:"))
    async def adm_ok(cq: CallbackQuery):
        if not is_admin(cfg, cq.from_user.id):
            await cq.answer("Нет доступа", show_alert=True)
            return
        order_id = int(cq.data.split(":", 1)[1])
        order = await db_get(order_id)
        if not order:
            await cq.answer("Заявка не найдена", show_alert=True)
            return
        await db_set_status(order_id, "approved", admin_id=cq.from_user.id)
        await cq.message.edit_text(f"✅ Подтверждено: #{order_id}\n(Выдача на сервер подключится позже.)")
        await cq.answer("Ок")

    @dp.callback_query(F.data.startswith("adm_no:"))
    async def adm_no(cq: CallbackQuery):
        if not is_admin(cfg, cq.from_user.id):
            await cq.answer("Нет доступа", show_alert=True)
            return
        order_id = int(cq.data.split(":", 1)[1])
        order = await db_get(order_id)
        if not order:
            await cq.answer("Заявка не найдена", show_alert=True)
            return
        await cq.message.edit_text(admin_card(order) + "\n\n❌ <b>Выбери причину отказа:</b>", reply_markup=kb_reasons(order_id))
        await cq.answer()

    @dp.callback_query(F.data.startswith("adm_reason:"))
    async def adm_reason(cq: CallbackQuery):
        if not is_admin(cfg, cq.from_user.id):
            await cq.answer("Нет доступа", show_alert=True)
            return
        _, order_id_str, tag = cq.data.split(":", 2)
        order_id = int(order_id_str)
        order = await db_get(order_id)
        if not order:
            await cq.answer("Заявка не найдена", show_alert=True)
            return

        if tag == "back":
            await cq.message.edit_text(admin_card(order), reply_markup=kb_admin(order_id))
            await cq.answer()
            return

        reason_map = {
            "sum": "Неверная сумма",
            "code": "Нет кода в начале комментария",
            "receipt": "Чек не читается или не приложен",
            "dup": "Уже выдано ранее",
            "fake": "Подозрение на подделку",
            "other": "Другая причина",
        }
        reason = reason_map.get(tag, "Отказ")
        await db_set_status(order_id, "rejected", admin_id=cq.from_user.id, reason=reason)

        await cq.message.edit_text(f"❌ Отклонено: #{order_id}\nПричина: <b>{reason}</b>")
        await cq.answer("Отклонено")

        try:
            await bot.send_message(order["tg_user_id"], f"❌ Оплата не подтверждена.\nПричина: <b>{reason}</b>")
        except Exception:
            pass

    # --- run polling with graceful shutdown ---
    polling_task = asyncio.create_task(dp.start_polling(bot))

    await stop_event.wait()

    polling_task.cancel()
    try:
        await polling_task
    except Exception:
        pass

    try:
        await http_runner.cleanup()
    except Exception:
        pass

    await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())
