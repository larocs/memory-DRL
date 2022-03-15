import setuptools

print('Installing packages:', setuptools.find_packages())

with open('requirements.txt') as reqs_file:
    install_requires = [
        line
        for line in reqs_file
    ]

setuptools.setup(
    name="sac-experiments",
    version="0.0.1",
    author="Samuel Chenatti",
    author_email="samuel.chenatti@gmail.com",
    description="SAC implementation for experimentation with new models",
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
    install_requires=install_requires
)
