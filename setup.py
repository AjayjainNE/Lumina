from setuptools import setup, find_packages

setup(
    name="lumina-doc-intelligence",
    version="1.0.0",
    description="Multi-Agent Document Intelligence with RL-Optimised Routing & LLM Self-Calibration",
    author="LUMINA Project",
    python_requires=">=3.10",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.40.0",
        "sentence-transformers>=2.7.0",
        "stable-baselines3>=2.3.0",
        "gymnasium>=0.29.0",
        "fastapi>=0.111.0",
        "uvicorn[standard]>=0.29.0",
        "httpx>=0.27.0",
        "pydantic>=2.7.0",
        "mlflow>=2.12.0",
        "numpy>=1.26.0",
        "pandas>=2.2.0",
        "python-dotenv>=1.0.0",
    ],
    extras_require={
        "dashboard": ["streamlit>=1.35.0", "plotly>=5.22.0"],
        "interpretability": ["shap>=0.45.0"],
        "tts": ["gTTS>=2.5.1"],
        "pdf": ["pdfplumber>=0.10.3"],
        "dev": ["pytest>=8.2.0", "pytest-asyncio>=0.23.0", "ruff>=0.4.0", "black>=24.0.0"],
    },
    entry_points={
        "console_scripts": [
            "lumina-api=api.app:main",
        ],
    },
    classifiers=[
    ],
)
