from setuptools import find_packages, setup

setup(
    name='fraud_detection',
    packages=find_packages(),
    version='0.1.0',
    description='A robust fraud detection pipeline with advanced feature engineering and model ensemble',
    author='Your Name',
    license='MIT',
    install_requires=[
        'numpy>=1.24.3',
        'pandas>=2.0.3',
        'scikit-learn>=1.3.0',
        'xgboost>=2.0.0',
        'lightgbm>=4.1.0',
        'imbalanced-learn>=0.11.0',
        'matplotlib>=3.7.2',
        'seaborn>=0.12.2',
        'joblib>=1.3.2',
        'python-dotenv>=1.0.0'
    ],
    python_requires='>=3.8',
) 