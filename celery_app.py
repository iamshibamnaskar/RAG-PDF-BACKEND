# celery_app.py
from celery import Celery

celery_app = Celery(
    "pdf_tasks",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/1",
    include=["tasks"]
)

# Optional: some sane defaults
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
)
