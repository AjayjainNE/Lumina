"""LUMINA pytest configuration."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("MISTRAL_API_KEY", "Insert your own API key")
os.environ.setdefault("MLFLOW_TRACKING_URI", "http://localhost:5000")
