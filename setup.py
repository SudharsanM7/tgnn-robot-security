from setuptools import setup, find_packages

setup(
    name="tgnn-adversarial-detection",
    version="1.0.0",
    author="Temporal GNN Research Team",
    description="Temporal Graph Neural Network with Trust Propagation for Adversarial Attack Detection in Multi-Robot Communication Networks",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torch-geometric>=2.4.0",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "pandas>=2.1.0",
        "networkx>=3.2",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.8.0",
        "pyyaml>=6.0",
        "tqdm>=4.66.0",
        "tensorboard>=2.15.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.12.0",
            "flake8>=6.1.0",
        ]
    }
)
