"""Event bus.

Hackathon build: an in-process asyncio fan-out (one queue per subscriber,
e.g. each dashboard SSE connection).

Production design: the same `publish()` call also produces to Kafka topic
`shelf.events` (keyed by shelf_id so per-bay ordering holds). Enable with
SHELFSENSE_KAFKA=1 and KAFKA_BOOTSTRAP=host:9092 (requires `pip install aiokafka`).
"""
from __future__ import annotations

import asyncio
import json
import logging
import os

log = logging.getLogger("shelfsense.bus")


class KafkaAdapter:  # ---- KAFKA PRODUCER ADAPTER (optional, production path) ----
    def __init__(self, bootstrap: str, topic: str = "shelf.events"):
        self.bootstrap, self.topic, self.producer = bootstrap, topic, None

    async def start(self):
        from aiokafka import AIOKafkaProducer  # lazy: only needed when enabled
        self.producer = AIOKafkaProducer(bootstrap_servers=self.bootstrap,
                                         value_serializer=lambda v: json.dumps(v).encode())
        await self.producer.start()
        log.info("Kafka producer connected to %s", self.bootstrap)

    async def send(self, msg: dict):
        key = str(msg.get("event", {}).get("shelf_id", "")).encode()
        await self.producer.send_and_wait(self.topic, msg, key=key)

    async def stop(self):
        if self.producer:
            await self.producer.stop()


class EventBus:
    def __init__(self):
        self.subscribers: set[asyncio.Queue] = set()
        self.kafka: KafkaAdapter | None = None

    async def start(self):
        if os.getenv("SHELFSENSE_KAFKA") == "1":
            self.kafka = KafkaAdapter(os.getenv("KAFKA_BOOTSTRAP", "localhost:9092"))
            try:
                await self.kafka.start()
            except Exception as e:  # keep the demo alive without a broker
                log.warning("Kafka disabled (%s); using in-process bus only", e)
                self.kafka = None

    async def stop(self):
        if self.kafka:
            await self.kafka.stop()

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=1000)
        self.subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue):
        self.subscribers.discard(q)

    async def publish(self, kind: str, event: dict):
        msg = {"kind": kind, "event": event}
        for q in list(self.subscribers):
            if not q.full():
                q.put_nowait(msg)
        if self.kafka:
            await self.kafka.send(msg)
