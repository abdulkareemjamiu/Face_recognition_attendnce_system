import os
from pathlib import Path

# Security
SECRET_KEY = os.urandom(32)

# Database configuration
BASE_DIR = Path(__file__).parent
SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or f'sqlite:///{BASE_DIR}/database.db'
SQLALCHEMY_TRACK_MODIFICATIONS = False

# Ensure migrations work properly
MIGRATIONS_DIR = BASE_DIR / 'migrations'



