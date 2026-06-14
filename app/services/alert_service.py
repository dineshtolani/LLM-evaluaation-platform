import httpx
import json
import logging
from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.models.alert import Alert, AlertStatus, AlertMetric, AlertOperator

logger = logging.getLogger(__name__)


class AlertService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def check_alerts(self, metric: AlertMetric, value: float, prompt_id=None, llm_model_id=None):
        query = select(Alert).where(
            Alert.metric == metric,
            Alert.status == AlertStatus.active,
        )
        result = await self.db.execute(query)
        alerts = result.scalars().all()

        triggered = []
        for alert in alerts:
            if alert.prompt_id and alert.prompt_id != prompt_id:
                continue
            if alert.llm_model_id and alert.llm_model_id != llm_model_id:
                continue

            if self._evaluate_condition(value, alert.operator, alert.threshold):
                alert.status = AlertStatus.triggered
                alert.last_triggered_at = datetime.now(timezone.utc)
                triggered.append(alert)

                if alert.notification_url:
                    await self._send_webhook(alert, value)

        if triggered:
            await self.db.commit()

        return triggered

    def _evaluate_condition(self, value: float, operator: AlertOperator, threshold: float) -> bool:
        ops = {
            AlertOperator.gt: lambda v, t: v > t,
            AlertOperator.lt: lambda v, t: v < t,
            AlertOperator.gte: lambda v, t: v >= t,
            AlertOperator.lte: lambda v, t: v <= t,
            AlertOperator.eq: lambda v, t: v == t,
        }
        return ops[operator](value, threshold)

    async def _send_webhook(self, alert: Alert, value: float):
        try:
            async with httpx.AsyncClient() as client:
                payload = {
                    "alert_id": str(alert.id),
                    "alert_name": alert.name,
                    "metric": alert.metric.value,
                    "threshold": alert.threshold,
                    "current_value": value,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                await client.post(alert.notification_url, json=payload, timeout=10)
        except Exception as e:
            logger.error(f"Failed to send webhook for alert {alert.id}: {e}")
