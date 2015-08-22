from setuptools import setup

with open('README.rst') as file:
    long_description = file.read()

setup(
    name='simple_spearmint',
    version='0.0.0',
    description='Thin wrapper class around spearmint',
    author='Colin Raffel',
    author_email='craffel@gmail.com',
    url='https://github.com/craffel/simple_spearmint',
    packages=['simple_spearmint'],
    long_description=long_description,
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        'Development Status :: 5 - Production/Stable',
        "Intended Audience :: Developers"
    ],
    keywords='hyperparameter optimization',
    license='MIT',
    install_requires=[
        'spearmint',
        'numpy',
    ],
)
