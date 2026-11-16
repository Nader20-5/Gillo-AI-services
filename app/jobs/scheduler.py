"""
Scheduler setup for background jobs in Gillo AI Service.
Switchable via settings.ENABLE_BACKGROUND_JOBS.
"""
import logging
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

from app.config import settings
from app.database import get_database
from app.jobs.sprint_jobs import (
    run_daily_risk_scan,
    expire_and_supersede_suggestions,
    refresh_developer_ratios
)

logger = logging.getLogger(__name__)

_scheduler: AsyncIOScheduler | None = None


def start_scheduler():
    """Initializes and starts background jobs scheduler if enabled."""
    global _scheduler
    if not settings.ENABLE_BACKGROUND_JOBS:
        logger.info("Background jobs scheduler is disabled by configuration (ENABLE_BACKGROUND_JOBS=False).")
        return

    if _scheduler is not None and _scheduler.running:
        return

    _scheduler = AsyncIOScheduler()

    async def _scheduled_risk_scan():
        db = get_database()
        res = await run_daily_risk_scan(db)
        logger.info(f"Completed scheduled daily risk scan: {res}")

    async def _scheduled_expire_and_supersede():
        db = get_database()
        exp, sup = await expire_and_supersede_suggestions(db)
        if exp > 0 or sup > 0:
            logger.info(f"Completed suggestions housekeeping: {exp} expired, {sup} superseded.")

    async def _scheduled_ratio_refresh():
        db = get_database()
        cnt = await refresh_developer_ratios(db)
        logger.info(f"Completed scheduled ratio refresh for {cnt} developers.")

    # Schedule risk scan daily (every 24 hours), expire checks hourly, and ratio refresh daily
    _scheduler.add_job(_scheduled_risk_scan, IntervalTrigger(hours=24), id="daily_risk_scan", replace_existing=True)
    _scheduler.add_job(_scheduled_expire_and_supersede, IntervalTrigger(hours=1), id="expire_housekeeping", replace_existing=True)
    _scheduler.add_job(_scheduled_ratio_refresh, IntervalTrigger(hours=24), id="ratio_refresh", replace_existing=True)

    _scheduler.start()
    logger.info("Background jobs scheduler started successfully.")


def stop_scheduler():
    """Shuts down the background scheduler."""
    global _scheduler
    if _scheduler and _scheduler.running:
        logger.info("Shutting down background scheduler...")
        _scheduler.shutdown(wait=False)
        _scheduler = None
