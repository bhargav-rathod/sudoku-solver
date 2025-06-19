# gunicorn_config.py
import multiprocessing
import os

# Server socket
bind = f"0.0.0.0:{os.environ.get('PORT', 5000)}"
backlog = 2048

# Worker processes
workers = min(multiprocessing.cpu_count() * 2 + 1, 4)  # Limit workers to prevent memory issues
worker_class = "sync"
worker_connections = 1000
timeout = 60  # Increased timeout for image processing
keepalive = 2

# Restart workers after this many requests to prevent memory leaks
max_requests = 100
max_requests_jitter = 10

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(T)s'

# Process naming
proc_name = 'sudoku_solver'

# Server mechanics
daemon = False
pidfile = None
user = None
group = None
tmp_upload_dir = None

# SSL
keyfile = None
certfile = None

# Memory management
preload_app = True
worker_tmp_dir = "/dev/shm"  # Use memory for temporary files if available

# Graceful timeout
graceful_timeout = 30

# Security
limit_request_line = 4094
limit_request_fields = 100
limit_request_field_size = 8190