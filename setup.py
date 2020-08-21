import setuptools
import tf_lightning

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tf_nightly-7vasudevgupta",
    version=tf_lightning.__version__,
    author=tf_lightning.__author__ ,
    author_email=tf_lightning.__author_email__,
    description="Light wrapper for models training in tf2",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="Apache",
    url="https://github.com/VasudevGupta7/tf-lightning",
    packages=setuptools.find_packages(),
    install_requires=[
        "pathlib==1.0.1",
        "wandb==0.9.4",
        # "tensorflow==2.3"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],
    python_requires='>=3.7',
)