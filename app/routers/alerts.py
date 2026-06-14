from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from typing import Optional
import uuid

from app.database import get_db
from app.models.alert import Alert
from app.schemas.alert import AlertCreate, AlertUpdate, AlertResponse, AlertList

router = APIRouter(prefix="/api/alerts", tags=["alerts"])


@router.post("", response_model=AlertResponse, status_code=201)
async def create_alert(alert_data: AlertCreate, db: AsyncSession = Depends(get_db)):
    alert = Alert(**alert_data.model_dump())
    db.add(alert)
    await db.commit()
    await db.refresh(alert)
    return alert


@router.get("", response_model=AlertList)
async def list_alerts(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    status: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    query = select(Alert)
    if status:
        query = query.where(Alert.status == status)

    count_query = select(func.count()).select_from(query.subquery())
    total = await db.scalar(count_query)

    query = query.offset((page - 1) * page_size).limit(page_size).order_by(Alert.created_at.desc())
    result = await db.execute(query)
    alerts = result.scalars().all()
    return AlertList(total=total, items=alerts)


@router.get("/{alert_id}", response_model=AlertResponse)
async def get_alert(alert_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Alert).where(Alert.id == alert_id))
    alert = result.scalar_one_or_none()
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    return alert


@router.put("/{alert_id}", response_model=AlertResponse)
async def update_alert(alert_id: uuid.UUID, alert_data: AlertUpdate, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Alert).where(Alert.id == alert_id))
    alert = result.scalar_one_or_none()
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")

    for key, value in alert_data.model_dump(exclude_unset=True).items():
        setattr(alert, key, value)

    await db.commit()
    await db.refresh(alert)
    return alert


@router.delete("/{alert_id}", status_code=204)
async def delete_alert(alert_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Alert).where(Alert.id == alert_id))
    alert = result.scalar_one_or_none()
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    await db.delete(alert)
    await db.commit()
