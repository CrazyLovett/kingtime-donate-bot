import asyncio
import os
import random
import re
import signal
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
        cfg = yaml.safe_load(f)

    # secrets from ENV only
    cfg["bot"]["token"] = env("BOT_TOKEN")
    cfg["api"]["token"] = env("API_TOKEN")
    cfg["payment"]["card"] = env("CARD_NUMBER")

    # optional, for future webhook
    cfg["app"]["public_url"] = env("PUBLIC_URL", cfg["app"].get("public_url", ""))

    if not cfg["bot"]["token"]:
        raise RuntimeError("ENV BOT_TOKEN not set")
    if not cfg["api"]["token"]:
        raise RuntimeError("ENV API_TOKEN not set")
    if not cfg["payment"]["card"]:
        raise RuntimeError("ENV CARD_NUMBER not set")

    return cfg


# ===================== STATES =====================
class BuyFlow(StatesGroup):
    choosing_category = State()
    choosing_product = State()
    entering_nick = State()
    waiting_receipt = State()


# ===================== PRODUCT HELPERS =====================
def list_products(cfg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return cfg.get("products", {})


def product_categories(cfg: Dict[str, Any]) -> List[str]:
    # categories are values: donate/case/service
    cats = set()
    for p in list_products(cfg).values():
        cats.add(p.get("type", "other"))
    order = ["donate", "case", "service", "other"]
    return [c for c in order if c in cats] + [c for c in sorted(cats) if c not in order]


def products_in_category(cfg: Dict[str, Any], cat: str) -> List[tuple[str, Dict[str, Any]]]:
    items = [(k, v) for k, v in list_products(cfg).items() if v.get("type", "other") == cat]
    items.sort(key=lambda kv: (int(kv[1].get("price", 0)), str(kv[1].get("title", "")).lower()))
    return items


def gen_code(prefix: str, length: int) -> str:
    chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    return prefix + "-" + "".join(random.choice(chars) for _ in range(length))


def pretty_card(raw: str) -> str:
    raw = (raw or "").strip().replace(" ", "")
    return " ".join(raw[i:i+4] for i in range(0, len(raw), 4))


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
        f"• <b>Получатель:</b> {p['fio']}\n"
        + (f"• <b>Банк:</b> {bank}\n" if bank else "")
        + f"• <b>Номер карты:</b> <code>{pretty_card(p['card'])}</code>\n\n"
        "✅ После оплаты нажми <b>«Я оплатил»</b> и пришли <b>чек/скрин</b>."
    )


# ===================== KEYBOARDS =====================
def kb_categories(cfg: Dict[str, Any]) -> InlineKeyboardMarkup:
    cat_names = {
        "donate": "💎 Донаты",
        "case": "🎁 Кейсы",
        "service": "🛠 Услуги",
        "other": "📦 Другое",
    }
    rows = []
    for c in product_categories(cfg):
        rows.append([InlineKeyboardButton(text=cat_names.get(c, c), callback_data=f"cat:{c}")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def kb_products(cfg: Dict[str, Any], cat: str) -> InlineKeyboardMarkup:
    rows = []
    for key, p in products_in_category(cfg, cat):
        rows.append([InlineKeyboardButton(
            text=f"{p.get('title','')} — {int(p.get('price',0))}₽",
            callback_data=f"buy:{key}"
        )])
    rows.append([InlineKeyboardButton(text="⬅️ Назад", callback_data="back:cats")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def kb_after_pay(order_id: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="✅ Я оплатил (отправить чек)", callback_data=f"paid:{order_id}")],
        [InlineKeyboardButton(text="⬅️ В магазин", callback_data="back:cats")],
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
        inline_keyboard=[[InlineKeyboardButton(text=t, callback_data=f"adm_reason:{order_id}:{tag}")]
                         for (t, tag) in reasons]
    )


# ===================== DB MIGRATION =====================
EXPECTED_COLUMNS = [
    "id",
    "created_at",
    "tg_user_id",
    "tg_username",
    "nick",
    "product_key",
    "product_title",
    "amount_rub",
    "code",
    "status",            # waiting_receipt / pending_review / approved / issuing / issued / rejected / failed
    "receipt_file_id",
    "admin_id",
    "reject_reason",
    "issued_at",
]

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
  receipt_file_id TEXT,
  admin_id INTEGER,
  reject_reason TEXT,
  issued_at TEXT
)
"""


async def get_columns(db: aiosqlite.Connection, table: str) -> List[str]:
    cur = await db.execute(f"PRAGMA table_info({table})")
    rows = await cur.fetchall()
    return [r[1] for r in rows]  # column name is at index 1


async def db_init():
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(CREATE_SQL)
        await db.commit()

        cols = await get_columns(db, "orders")
        if cols == EXPECTED_COLUMNS:
            return

        # If mismatch -> migrate safely
        await db.execute("ALTER TABLE orders RENAME TO orders_old")
        await db.execute(CREATE_SQL)

        old_cols = await get_columns(db, "orders_old")
        common = [c for c in EXPECTED_COLUMNS if c in old_cols]

        if common:
            common_csv = ",".join(common)
            await db.execute(
                f"INSERT INTO orders ({common_csv}) SELECT {common_csv} FROM orders_old"
            )
        await db.execute("DROP TABLE orders_old")
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


async def db_set_receipt(order_id: int, file_id: str):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE orders SET receipt_file_id=?, status=? WHERE id=?",
            (file_id, "pending_review", order_id),
        )
        await db.commit()


async def db_set_status(order_id: int, status: str, admin_id: Optional[int] = None, reason: Optional[str] = None):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """
            UPDATE orders
            SET status=?,
                admin_id=COALESCE(?, admin_id),
                reject_reason=COALESCE(?, reject_reason)
            WHERE id=?
            """,
            (status, admin_id, reason, order_id),
        )
        await db.commit()


# ===================== ADMIN CARD =====================
def admin_card(order: dict) -> str:
    return (
        "💰 <b>Заявка на донат</b>\n\n"
        f"🆔 <b>Заявка:</b> #{order['id']}\n"
        f"👤 <b>Ник:</b> <code>{order['nick']}</code>\n"
        f"📦 <b>Товар:</b> {order['product_title']}\n"
        f"💵 <b>Сумма:</b> {order['amount_rub']} ₽\n"
        f"🏷 <b>Код:</b> <code>{order['code']}</code>\n"
        f"🙋 <b>TG:</b> {order['tg_user_id']}" + (f" (@{order['tg_username']})" if order.get("tg_username") else "")
    )


# ===================== HTTP API (for plugin) =====================
async def http_index(_req: web.Request) -> web.Response:
    return web.Response(text="OK", content_type="text/plain")


async def api_pull(req: web.Request) -> web.Response:
    cfg: Dict[str, Any] = req.app["cfg"]
    token = req.query.get("token", "")
    if token != cfg["api"]["token"]:
        return web.json_response({"ok": False, "error": "unauthorized"}, status=401)

    limit = int(req.query.get("limit", "10"))
    limit = max(1, min(limit, 50))

    out = []
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        await db.execute("BEGIN IMMEDIATE")

        cur = await db.execute(
            "SELECT * FROM orders WHERE status='approved' ORDER BY id ASC LIMIT ?",
            (limit,),
        )
        rows = await cur.fetchall()
        ids = [r["id"] for r in rows]

        if ids:
            qmarks = ",".join(["?"] * len(ids))
            await db.execute(f"UPDATE orders SET status='issuing' WHERE id IN ({qmarks})", ids)

        await db.commit()

        for r in rows:
            prod = cfg["products"].get(r["product_key"], {})
            cmds = [str(c).format(nick=r["nick"]) for c in prod.get("commands", [])]
            out.append({
                "orderId": r["id"],
                "nick": r["nick"],
                "productKey": r["product_key"],
                "productTitle": r["product_title"],
                "amountRub": r["amount_rub"],
                "announceItem": prod.get("announce", r["product_title"]),
                "commands": cmds,
                "tgUserId": r["tg_user_id"],
            })

    return web.json_response({"ok": True, "orders": out})


async def api_ack(req: web.Request) -> web.Response:
    cfg: Dict[str, Any] = req.app["cfg"]
    bot: Bot = req.app["bot"]

    try:
        data = await req.json()
    except Exception:
        return web.json_response({"ok": False, "error": "bad_json"}, status=400)

    if data.get("token") != cfg["api"]["token"]:
        return web.json_response({"ok": False, "error": "unauthorized"}, status=401)

    order_id = int(data.get("orderId", 0))
    ok = bool(data.get("ok", False))
    error = str(data.get("error", ""))[:500]

    order = await db_get(order_id)
    if not order:
        return web.json_response({"ok": False, "error": "order_not_found"}, status=404)

    if ok:
        issued_at = now_iso()
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute("UPDATE orders SET status='issued', issued_at=? WHERE id=?", (issued_at, order_id))
            await db.commit()
        try:
            await bot.send_message(
                order["tg_user_id"],
                f"🎉 <b>Покупка выдана!</b>\n"
                f"📦 {order['product_title']}\n"
                f"👤 <code>{order['nick']}</code>\n\nСпасибо за поддержку ❤️"
            )
        except Exception:
            pass
    else:
        await db_set_status(order_id, "failed", reason=error or "server_error")
        try:
            await bot.send_message(
                order["tg_user_id"],
                f"⚠️ <b>Не получилось выдать покупку</b>\n"
                f"📦 {order['product_title']}\n"
                f"👤 <code>{order['nick']}</code>\n\n"
                f"Ошибка: <code>{error or 'server_error'}</code>\n"
                "Напиши администрации — исправим."
            )
        except Exception:
            pass

    return web.json_response({"ok": True})


# ===================== MAIN =====================
async def main():
    cfg = load_cfg()
    await db_init()

    bot = Bot(cfg["bot"]["token"], default=DefaultBotProperties(parse_mode=ParseMode.HTML))
    dp = Dispatcher()

    # ----- HTTP server for Render + plugin API -----
    http_app = web.Application()
    http_app["cfg"] = cfg
    http_app["bot"] = bot
    http_app.router.add_get("/", http_index)
    http_app.router.add_get("/api/pull", api_pull)
    http_app.router.add_post("/api/ack", api_ack)

    port = int(env("PORT", "10000"))
    runner = web.AppRunner(http_app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()
    print(f"HTTP started on 0.0.0.0:{port}")

    # ----- BOT HANDLERS -----
    @dp.message(CommandStart())
    async def start_cmd(m: Message, state: FSMContext):
        await state.clear()
        await m.answer("🛒 <b>KingTime | Донат магазин</b>\nВыбери категорию:", reply_markup=kb_categories(cfg))

    @dp.message(Command("donate"))
    async def donate_cmd(m: Message, state: FSMContext):
        await state.clear()
        await m.answer("🛒 Категории:", reply_markup=kb_categories(cfg))

    @dp.callback_query(F.data == "back:cats")
    async def back_cats(cq: CallbackQuery, state: FSMContext):
        await state.clear()
        await cq.message.edit_text("🛒 Выбери категорию:", reply_markup=kb_categories(cfg))
        await cq.answer()

    @dp.callback_query(F.data.startswith("cat:"))
    async def cat_pick(cq: CallbackQuery, state: FSMContext):
        cat = cq.data.split(":", 1)[1]
        await state.set_state(BuyFlow.choosing_product)
        await state.update_data(category=cat)
        await cq.message.edit_text("📦 Выбери товар:", reply_markup=kb_products(cfg, cat))
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
        await cq.message.edit_text(f"✍️ Введи ник для <b>{p['title']}</b>:")
        await cq.answer()

    @dp.message(BuyFlow.entering_nick)
    async def got_nick(m: Message, state: FSMContext):
        nick = (m.text or "").strip()
        if not NICK_RE.match(nick):
            await m.answer("❌ Ник неверный. Пример: <code>Steve_123</code>\nПопробуй снова:")
            return

        data = await state.get_data()
        key = data["product_key"]
        p = cfg["products"].get(key)
        if not p:
            await m.answer("❌ Товар не найден. Вернись в магазин: /start")
            await state.clear()
            return

        code = gen_code(cfg["payment"]["comment_prefix"], int(cfg["payment"]["code_length"]))
        order_id = await db_create_order(
            tg_user_id=m.from_user.id,
            tg_username=m.from_user.username,
            nick=nick,
            product_key=key,
            product_title=str(p["title"]),
            amount_rub=int(p["price"]),
            code=code,
        )
        await state.clear()
        await m.answer(payment_text(cfg, nick, str(p["title"]), int(p["price"]), code), reply_markup=kb_after_pay(order_id))

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
        await m.answer("✅ Принято! Отправлено на проверку админу.")

        for admin_id in cfg["admins"]:
            try:
                await bot.send_message(admin_id, admin_card(await db_get(order_id)), reply_markup=kb_admin(order_id))
                await bot.send_photo(admin_id, file_id, caption=f"📎 Чек к заявке #{order_id}")
            except Exception:
                pass

    @dp.message(BuyFlow.waiting_receipt)
    async def receipt_wrong(m: Message):
        await m.answer("Пришли именно <b>фото</b> или <b>документ</b> с чеком/скрином.")

    @dp.callback_query(F.data.startswith("adm_ok:"))
    async def adm_ok(cq: CallbackQuery):
        if cq.from_user.id not in set(cfg["admins"]):
            await cq.answer("Нет доступа", show_alert=True)
            return
        order_id = int(cq.data.split(":", 1)[1])
        order = await db_get(order_id)
        if not order:
            await cq.answer("Заявка не найдена", show_alert=True)
            return
        await db_set_status(order_id, "approved", admin_id=cq.from_user.id)
        await cq.message.edit_text(f"✅ Подтверждено: #{order_id}\nСервер выдаст автоматически.")
        await cq.answer("Ок")

    @dp.callback_query(F.data.startswith("adm_no:"))
    async def adm_no(cq: CallbackQuery):
        if cq.from_user.id not in set(cfg["admins"]):
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
        if cq.from_user.id not in set(cfg["admins"]):
            await cq.answer("Нет доступа", show_alert=True)
            return
        _, oid, tag = cq.data.split(":", 2)
        order_id = int(oid)
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

    # ----- graceful shutdown -----
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
