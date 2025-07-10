# setup.py
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="llm-dev-assistant",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="An automated development assistant using LLMs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/llm-dev-assistant",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Code Generators",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.7",
    install_requires=[
        "openai>=0.27.0",
        "pygithub>=2.1.1",
        "gitpython>=3.1.40",
        "requests>=2.25.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=22.0",
            "flake8>=4.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "llm-dev-assistant=llm_dev_assistant.__main__:main",
            "llm-dev-gui=llm_dev_assistant.gui.main_window:main",
        ],
    },
    include_package_data=True,
    package_data={
        "llm_dev_assistant": ["*.txt", "*.md"],
    },
)