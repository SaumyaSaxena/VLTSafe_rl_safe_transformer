from setuptools import find_packages, setup
setup(
    name='VLTSafe_rl_safe_transformer',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        "imageio",
        "omegaconf",
        "wandb",
    ],
    include_package_data=True,
)