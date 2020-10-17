import setuptools

print('Installing packages:', setuptools.find_packages())

setuptools.setup(
    name="sac-experiments",
    version="0.0.1",
    author="Samuel Chenatti",
    author_email="samuel.chenatti@gmail.com",
    description="SAC implementation for experimentation with new models",
    packages=setuptools.find_packages(),
    python_requires='>=3.6'
)
