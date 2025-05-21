from setuptools import setup, find_packages

setup(
    name="limitor",
    version="0.1",
    packages=find_packages(),
    install_requires=["click","numpy", "matplotlib", "scipy","torch"],
    description="FC Method",
    entry_points = dict(
        console_scripts = [
            'limitor = limitor.__main__:main',
        ]
    ),
)
