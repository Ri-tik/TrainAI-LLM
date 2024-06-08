from setuptools import setup, find_packages

setup(
    name='trainaillm',
    version='0.1',
    packages=find_packages(),
    description='llm for training your own ai',
    long_description=open('README.md').read(),
    author='Ritik',
    url='https://github.com/Ri-tik/TrainAILLM',
    package_data={
        '': ['*.txt', '*.md', '*.json'],  # Add other file types as needed.
        'trainaillm': ['data/*', 'data/*/*', 'data/*/*/*'],  # Assuming 'data' is directly inside the 'trainaillm' package.
    },
    include_package_data=True,
)
