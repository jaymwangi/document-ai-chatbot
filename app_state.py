# app_state.py - COMPLETE FIXED VERSION with All Initializations

import streamlit as st
import time
import uuid
import hashlib
import logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class DocumentMetadata:
    doc_id: str
    filename: str
    chunks: int = 0
    vectors: int = 0
    processed_at: float = field(default_factory=time.time)


class AppState:
    """Clean Event-Loop AppState - Single Source of Truth"""
    
    def __init__(self):
        self._init()
    
    @staticmethod
    def _init():
        """Initialize ALL session state variables."""
        
        # === CORE STATE ===
        if "pipeline" not in st.session_state:
            st.session_state.pipeline = None
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "documents" not in st.session_state:
            st.session_state.documents = {}
        if "current_doc" not in st.session_state:
            st.session_state.current_doc = None
        if "doc_count" not in st.session_state:
            st.session_state.doc_count = 0
        if "processed" not in st.session_state:
            st.session_state.processed = set()
        
        # === QUEUE ===
        if "queue" not in st.session_state:
            st.session_state.queue = []
        
        # === PROCESSING TRIGGER ===
        if "trigger_process" not in st.session_state:
            st.session_state.trigger_process = False
        
        # === LOCK ===
        if "processing" not in st.session_state:
            st.session_state.processing = None
        if "is_processing" not in st.session_state:
            st.session_state.is_processing = False
        
        # === HASH ===
        if "last_hash" not in st.session_state:
            st.session_state.last_hash = None
        
        # === RATE LIMIT ===
        if "rate_limit" not in st.session_state:
            st.session_state.rate_limit = {
                "remaining": 30,
                "reset_at": time.time() + 60,
                "last_request": 0,
                "min_interval": 3.0,
                "retry_count": 0,
            }
        else:
            # Ensure all keys exist
            rate = st.session_state.rate_limit
            if "min_interval" not in rate:
                rate["min_interval"] = 3.0
            if "retry_count" not in rate:
                rate["retry_count"] = 0
        
        # === PROCESSING STATUS ===
        if "processing_status" not in st.session_state:
            st.session_state.processing_status = {
                "is_running": False,
                "progress": 0.0,
                "stage": "idle",
                "stage_display": "📌 Ready",
                "details": "",
                "last_update": time.time(),
            }
        
        # === UI STATE ===
        if "initial_suggestions_shown" not in st.session_state:
            st.session_state.initial_suggestions_shown = False
        if "session_id" not in st.session_state:
            st.session_state.session_id = str(uuid.uuid4())
        
        # === EMBEDDER ===
        if "embedder_loaded" not in st.session_state:
            st.session_state.embedder_loaded = False
        if "embedder_load_time" not in st.session_state:
            st.session_state.embedder_load_time = None
        
        # === STATS ===
        if "total_requests" not in st.session_state:
            st.session_state.total_requests = 0
        if "total_errors" not in st.session_state:
            st.session_state.total_errors = 0
        if "last_processed_time" not in st.session_state:
            st.session_state.last_processed_time = None
    
    # ===== PROPERTIES =====
    @property
    def pipeline(self):
        return st.session_state.get("pipeline", None)
    
    @pipeline.setter
    def pipeline(self, value):
        st.session_state.pipeline = value
    
    @property
    def messages(self):
        return st.session_state.get("messages", [])
    
    @messages.setter
    def messages(self, value):
        st.session_state.messages = value
    
    @property
    def current_doc(self):
        return st.session_state.get("current_doc", None)
    
    @current_doc.setter
    def current_doc(self, value):
        st.session_state.current_doc = value
    
    @property
    def doc_count(self):
        return st.session_state.get("doc_count", 0)
    
    @doc_count.setter
    def doc_count(self, value):
        st.session_state.doc_count = value
    
    @property
    def is_ready(self) -> bool:
        return self.pipeline is not None and self.doc_count > 0
    
    @property
    def is_processing(self) -> bool:
        return st.session_state.get("is_processing", False)
    
    @is_processing.setter
    def is_processing(self, value: bool):
        st.session_state.is_processing = value
    
    @property
    def queue(self) -> List[Dict]:
        return st.session_state.get("queue", [])
    
    @property
    def current_doc_id(self):
        return st.session_state.get("current_doc", None)
    
    @current_doc_id.setter
    def current_doc_id(self, value):
        st.session_state.current_doc = value
    
    @property
    def document_count(self):
        return st.session_state.get("doc_count", 0)
    
    @document_count.setter
    def document_count(self, value):
        st.session_state.doc_count = value
    
    @property
    def processing_status(self):
        return st.session_state.get("processing_status", {
            "is_running": False,
            "progress": 0.0,
            "stage": "idle",
            "stage_display": "📌 Ready",
            "details": "",
            "last_update": time.time(),
        })
    
    @processing_status.setter
    def processing_status(self, value):
        st.session_state.processing_status = value
    
    @property
    def embedder_loaded(self):
        return st.session_state.get("embedder_loaded", False)
    
    @embedder_loaded.setter
    def embedder_loaded(self, value):
        st.session_state.embedder_loaded = value
    
    @property
    def embedder_load_time(self):
        return st.session_state.get("embedder_load_time", None)
    
    @embedder_load_time.setter
    def embedder_load_time(self, value):
        st.session_state.embedder_load_time = value
    
    @property
    def trigger_process(self) -> bool:
        return st.session_state.get("trigger_process", False)
    
    @trigger_process.setter
    def trigger_process(self, value: bool):
        st.session_state.trigger_process = value
    
    # ===== DOCUMENT METHODS =====
    def add_doc(self, doc_id: str, filename: str, chunks: int, vectors: int):
        st.session_state.documents[doc_id] = DocumentMetadata(
            doc_id=doc_id, filename=filename, chunks=chunks, vectors=vectors
        )
        st.session_state.current_doc = doc_id
        st.session_state.doc_count = len(st.session_state.documents)
    
    def add_document(self, doc_id: str, filename: str, chunks: int, vectors: int):
        self.add_doc(doc_id, filename, chunks, vectors)
    
    def mark_processed(self, file_id: str):
        st.session_state.processed.add(file_id)
    
    def is_processed(self, file_id: str) -> bool:
        return file_id in st.session_state.processed
    
    def is_file_processed(self, file_id: str) -> bool:
        return self.is_processed(file_id)
    
    def mark_file_processed(self, file_id: str):
        self.mark_processed(file_id)
    
    def get_document(self, doc_id: str) -> Optional[DocumentMetadata]:
        return st.session_state.documents.get(doc_id)
    
    def get_all_documents(self) -> List[DocumentMetadata]:
        return list(st.session_state.documents.values())
    
    # ===== QUEUE =====
    def enqueue(self, text: str) -> bool:
        h = hashlib.md5(text.encode()).hexdigest()
        
        if h == st.session_state.get("last_hash"):
            return False
        if any(q.get("hash") == h for q in self.queue):
            return False
        
        st.session_state.queue.append({
            "id": str(uuid.uuid4())[:8],
            "text": text,
            "hash": h,
            "timestamp": time.time(),
            "attempts": 0,
        })
        st.session_state.trigger_process = True
        return True
    
    def dequeue(self) -> Optional[Dict]:
        if not self.queue:
            return None
        return self.queue.pop(0)
    
    def queue_size(self) -> int:
        return len(self.queue)
    
    def clear_queue(self):
        st.session_state.queue = []
        st.session_state.trigger_process = False
    
    # ===== LOCK =====
    def lock(self, request_id: str, request_hash: str) -> bool:
        if st.session_state.get("processing") is not None:
            return False
        st.session_state.processing = request_id
        st.session_state.last_hash = request_hash
        st.session_state.is_processing = True
        logger.info(f"🔒 Lock acquired: {request_id}")
        return True
    
    def unlock(self):
        st.session_state.processing = None
        st.session_state.is_processing = False
        logger.info(f"🔓 Lock released, is_processing={st.session_state.is_processing}")
    
    # ===== RATE LIMITING =====
    def can_make_request(self) -> bool:
        rate = st.session_state.get("rate_limit", {})
        now = time.time()
        
        # Reset window if needed
        if now > rate.get("reset_at", now):
            rate["remaining"] = 30
            rate["reset_at"] = now + 60
            rate["retry_count"] = 0
            logger.info("🔄 Rate limit window reset")
            st.session_state.rate_limit = rate
        
        # Check min interval
        time_since_last = now - rate.get("last_request", 0)
        if time_since_last < rate.get("min_interval", 3.0):
            wait_time = rate.get("min_interval", 3.0) - time_since_last
            logger.debug(f"⏳ Min interval: {wait_time:.1f}s remaining")
            return False
        
        # Check remaining requests
        if rate.get("remaining", 0) <= 0:
            logger.debug(f"⏳ No requests remaining, reset in {rate.get('reset_at', now) - now:.0f}s")
            return False
        
        return True
    
    def consume_request(self) -> None:
        rate = st.session_state.get("rate_limit", {})
        rate["remaining"] = rate.get("remaining", 30) - 1
        rate["last_request"] = time.time()
        st.session_state.rate_limit = rate
        st.session_state.total_requests = st.session_state.get("total_requests", 0) + 1
        st.session_state.last_processed_time = time.time()
        logger.info(f"✅ Request consumed. Remaining: {rate['remaining']}")
    
    def get_rate_limit_status(self) -> Dict[str, Any]:
        rate = st.session_state.get("rate_limit", {})
        now = time.time()
        time_since_last = now - rate.get("last_request", 0)
        min_interval_remaining = max(0, rate.get("min_interval", 3.0) - time_since_last)
        
        return {
            "remaining": rate.get("remaining", 30),
            "reset_in": max(0, rate.get("reset_at", now + 60) - now),
            "can_request": self.can_make_request(),
            "min_interval_remaining": min_interval_remaining,
            "last_request": rate.get("last_request", 0),
            "total_requests": st.session_state.get("total_requests", 0),
            "total_errors": st.session_state.get("total_errors", 0),
        }
    
    def record_error(self):
        st.session_state.total_errors = st.session_state.get("total_errors", 0) + 1
    
    # ===== PROCESSING STATUS =====
    def start_processing(self):
        st.session_state.is_processing = True
        st.session_state.processing_status = {
            "is_running": True,
            "progress": 0.0,
            "stage": "starting",
            "stage_display": "🚀 Initializing...",
            "details": "Preparing to process documents...",
            "last_update": time.time(),
        }
    
    def update_processing_status(self, progress: float, stage: str, details: str = ""):
        stage_display_map = {
            "loading": "📄 Loading documents...",
            "chunking": "✂️ Chunking text...",
            "embedding": "🧠 Generating embeddings...",
            "indexing": "📊 Building index...",
            "finalizing": "✅ Finalizing...",
            "complete": "🎉 Complete!",
            "error": "❌ Error",
            "starting": "🚀 Initializing...",
        }
        status = st.session_state.processing_status
        status.update({
            "is_running": True,
            "progress": min(progress, 99.0),
            "stage": stage,
            "stage_display": stage_display_map.get(stage, stage),
            "details": details or status.get("details", ""),
            "last_update": time.time(),
        })
        st.session_state.processing_status = status
    
    def finish_processing(self, success: bool = True, error: Optional[str] = None):
        st.session_state.is_processing = False
        if success:
            st.session_state.processing_status = {
                "is_running": False,
                "progress": 100.0,
                "stage": "complete",
                "stage_display": "🎉 Complete!",
                "details": f"Successfully processed {self.doc_count} documents",
                "last_update": time.time(),
            }
        else:
            st.session_state.processing_status = {
                "is_running": False,
                "progress": 0.0,
                "stage": "error",
                "stage_display": f"❌ {error or 'Processing failed'}",
                "details": error or "Unknown error",
                "last_update": time.time(),
            }
            self.record_error()
    
    # ===== RESET =====
    def reset(self):
        st.session_state.pipeline = None
        st.session_state.messages = []
        st.session_state.documents = {}
        st.session_state.current_doc = None
        st.session_state.doc_count = 0
        st.session_state.processed = set()
        st.session_state.queue = []
        st.session_state.trigger_process = False
        st.session_state.processing = None
        st.session_state.last_hash = None
        st.session_state.is_processing = False
        st.session_state.processing_status = {
            "is_running": False,
            "progress": 0.0,
            "stage": "idle",
            "stage_display": "📌 Ready",
            "details": "",
            "last_update": time.time(),
        }
        st.session_state.rate_limit = {
            "remaining": 30,
            "reset_at": time.time() + 60,
            "last_request": 0,
            "min_interval": 3.0,
            "retry_count": 0,
        }
        st.session_state.embedder_loaded = False
        st.session_state.embedder_load_time = None
        st.session_state.total_requests = 0
        st.session_state.total_errors = 0
        st.session_state.last_processed_time = None


# ============================================================
# SINGLETON
# ============================================================

_app = None

def get_app_state():
    global _app
    if _app is None:
        _app = AppState()
    return _app


def get_state():
    return get_app_state()