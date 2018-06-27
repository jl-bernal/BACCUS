try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'description': 'MCMC using emcee software applying BACCUS formalism',
    'author': 'Jose Luis Bernal',
    'url': 'URL to get it at.',
    'author_email': 'joseluis.bernal@icc.ub.edu',
    'version': '0.1',
    'install_requires': ['nose','numpy','emcee','matplotlib','pymc3','theano'],
    'packages': ['baccus'],
    'name': 'baccus'
}

setup(**config)
