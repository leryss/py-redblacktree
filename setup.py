from setuptools import setup
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text('utf-8')

setup(
    name='redblacktree',
    version='1.0.2',    
    description='A pure python3 red black tree implementation',
	long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/leryss/redblacktree',
	download_url='https://github.com/leryss/py-redblacktree/archive/refs/tags/v1.0.tar.gz',
    author='leryss',
    license='MIT',
    packages=['redblacktree'],
    install_requires=[],

    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
)
