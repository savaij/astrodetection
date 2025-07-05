from setuptools import setup, find_packages

setup(
    name='astrodetection',
    version='0.1.0',
    description='Una libreria per analisi di dati',
    author='Il tuo nome',
    author_email='tuo@email.com',
    url='https://github.com/savaij/astrodetection',
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires='>=3.7',
)