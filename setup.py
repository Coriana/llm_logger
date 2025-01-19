from setuptools import setup, find_packages

setup(
    name='llm_logger',
    version='0.1.0',
    packages=find_packages(),
    description='A simple logger for Requests/Responses from LLMs',
    author='Corianas',
    author_email='corana@gmail.com',
    url='https://github.com/Coriana/llm_logger',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
