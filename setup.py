from setuptools import setup, find_packages

setup(
    name='h5visualizer',
    version='1.0.0',
    py_modules=['h5visualizer'],
    install_requires=[
        'tensorflow',
        'matplotlib',
        'numpy'
    ],
    entry_points={
        'console_scripts': [
            'h5visualizer=h5visualizer:main',  # Corrected entry point
        ],
    },
)
