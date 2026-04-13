from setuptools import setup, find_packages

setup(
    name='astrodetection',
    version='0.1.2',
    description='A Python library for detecting astroturfing (coordinated inauthentic behavior) in social media posts.',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    author='YOUR NAME',
    author_email='YOUR@EMAIL.COM',
    url='https://github.com/savaij/astrodetection',
    project_urls={
        'Bug Tracker': 'https://github.com/savaij/astrodetection/issues',
    },
    license='MIT',
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires='>=3.8',
    install_requires=[
        'pandas',
        'networkx',
        'tqdm',
        'demoji',
        'requests',
        'polyleven',
        'ipysigma',
        'numpy',
    ],
    extras_require={
        'standard': [
            'faiss-cpu',
            'fasttext',
            'gensim',
        ],
        'light': [
            'scikit-learn',
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Information Analysis',
    ],
)
