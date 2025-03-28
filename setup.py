from setuptools import setup, find_packages

setup(
    name="feapp-analysis",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "streamlit",
        "scipy",
        "pandas",
        "numpy",
        "matplotlib",
        "seaborn",
        "plotly",
        "feat",
        "opencv-python",
        "pillow",
    ],
    entry_points={
        "console_scripts": [
            "feapp=FEApp.FEApp:main",
        ],
    },
)