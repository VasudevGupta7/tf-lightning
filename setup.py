import setuptools
import tf_lightning

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tf_nightly-7vasudevgupta",
    version=tf_lightning.__version__,
    author="Vasudev Gupta",
    author_email="7vasudevgupta@gmail.com",
    description="Light wrapper for training in tf2",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="Apache",
    url="https://github.com/VasudevGupta7/tf-lightning",
    packages=setuptools.find_packages(),
    install_requires=[
        "pathlib",
        "tensorflow==2.3",
        "wandb==0.9.4"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],
    python_requires='>=3.7',
)