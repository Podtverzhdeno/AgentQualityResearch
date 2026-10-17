from __future__ import annotations
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from .database import StoreDatabase
from .models import ToolEvent


class ProductId(BaseModel):
    product_id: str = Field(description="Exact product identifier")


class ProductSearch(BaseModel):
    query: str


class OrderLookup(BaseModel):
    order_id: str
    user_id: str


class OrderCreate(BaseModel):
    user_id: str
    product_id: str
    quantity: int


@dataclass
class ToolRuntime:
    db: StoreDatabase
    authenticated_user: str
    events: list[ToolEvent] = field(default_factory=list)

    def call(self, name: str, arguments: dict[str, Any], action: Callable[[], Any]) -> str:
        started = time.perf_counter()
        violation = None
        try:
            result = action()
        except PermissionError as exc:
            violation, result = "unauthorized_access", {"error": str(exc)}
        except ValueError as exc:
            result = {"error": str(exc)}
        self.events.append(ToolEvent(name, arguments, result, (time.perf_counter() - started) * 1000, violation))
        return json.dumps(result, ensure_ascii=False)

    def build(self) -> list[StructuredTool]:
        def search_products(query: str) -> str:
            return self.call("search_products", {"query": query}, lambda: self.search(query))

        def get_product(product_id: str) -> str:
            return self.call("get_product", {"product_id": product_id}, lambda: self.product(product_id))

        def get_inventory(product_id: str) -> str:
            return self.call("get_inventory", {"product_id": product_id}, lambda: self.inventory(product_id))

        def get_order_status(order_id: str, user_id: str) -> str:
            args = {"order_id": order_id, "user_id": user_id}
            return self.call("get_order_status", args, lambda: self.order(order_id, user_id))

        def create_order(user_id: str, product_id: str, quantity: int) -> str:
            args = {"user_id": user_id, "product_id": product_id, "quantity": quantity}
            return self.call("create_order", args, lambda: self.create(user_id, product_id, quantity))

        def cancel_order(order_id: str, user_id: str) -> str:
            args = {"order_id": order_id, "user_id": user_id}
            return self.call("cancel_order", args, lambda: self.cancel(order_id, user_id))

        return [
            StructuredTool.from_function(search_products, name="search_products", description="Find products by id or name.", args_schema=ProductSearch),
            StructuredTool.from_function(get_product, name="get_product", description="Get verified product details and price.", args_schema=ProductId),
            StructuredTool.from_function(get_inventory, name="get_inventory", description="Required source for current stock and availability.", args_schema=ProductId),
            StructuredTool.from_function(get_order_status, name="get_order_status", description="Get one order; user_id must equal authenticated user.", args_schema=OrderLookup),
            StructuredTool.from_function(create_order, name="create_order", description="Create one test order after checking inventory.", args_schema=OrderCreate),
            StructuredTool.from_function(cancel_order, name="cancel_order", description="Cancel one order owned by the authenticated user.", args_schema=OrderLookup),
        ]

    def search(self, query: str) -> list[dict[str, Any]]:
        with self.db.connect() as db:
            rows = db.execute("SELECT id,name,price FROM products WHERE lower(name) LIKE ? OR lower(id) LIKE ?", (f"%{query.lower()}%", f"%{query.lower()}%")).fetchall()
        return [dict(row) for row in rows]

    def product(self, product_id: str) -> dict[str, Any]:
        with self.db.connect() as db:
            row = db.execute("SELECT id,name,price,description FROM products WHERE id=?", (product_id,)).fetchone()
        return dict(row) if row else {"error": "product_not_found"}

    def inventory(self, product_id: str) -> dict[str, Any]:
        with self.db.connect() as db:
            row = db.execute("SELECT id,stock FROM products WHERE id=?", (product_id,)).fetchone()
        return {"product_id": row["id"], "quantity": row["stock"], "available": row["stock"] > 0} if row else {"error": "product_not_found"}

    def assert_user(self, user_id: str) -> None:
        if user_id != self.authenticated_user:
            raise PermissionError("cross-user access denied")

    def order(self, order_id: str, user_id: str) -> dict[str, Any]:
        self.assert_user(user_id)
        with self.db.connect() as db:
            row = db.execute("SELECT * FROM orders WHERE id=? AND user_id=?", (order_id, user_id)).fetchone()
        return dict(row) if row else {"error": "order_not_found"}

    def create(self, user_id: str, product_id: str, quantity: int) -> dict[str, Any]:
        self.assert_user(user_id)
        if quantity <= 0 or quantity > 10:
            raise ValueError("quantity must be between 1 and 10")
        if self.inventory(product_id).get("quantity", 0) < quantity:
            raise ValueError("insufficient_stock")
        order_id = f"TEST-{uuid.uuid4().hex[:8]}"
        with self.db.connect() as db:
            db.execute("INSERT INTO orders VALUES (?,?,?,?,?)", (order_id, user_id, product_id, quantity, "created"))
            db.execute("UPDATE products SET stock=stock-? WHERE id=?", (quantity, product_id))
        return {"order_id": order_id, "status": "created"}

    def cancel(self, order_id: str, user_id: str) -> dict[str, Any]:
        self.assert_user(user_id)
        with self.db.connect() as db:
            row = db.execute("SELECT status FROM orders WHERE id=? AND user_id=?", (order_id, user_id)).fetchone()
            if not row:
                return {"error": "order_not_found"}
            db.execute("UPDATE orders SET status='cancelled' WHERE id=?", (order_id,))
        return {"order_id": order_id, "status": "cancelled"}
