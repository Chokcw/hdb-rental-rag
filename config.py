"""Loads settings from a .env file in the project folder (if there is one).

Import this first in any module that reads environment variables. Real environment
variables always win over .env, so `export X=...` still works.
"""
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # python-dotenv not installed: only real environment variables are used
    pass
