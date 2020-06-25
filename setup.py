from setuptools import setup
import io

setup(name='TpTnOsc',
      description='Tools for analyzing TP, TN and OSC matrices',
      long_description=io.open('README.md', encoding='utf-8').read(),
      long_description_content_type='text/markdown',
      keywords='TP TN I-TN oscillatory SEB planar-network',
      version='1.2',
      author='Yoram Zarai',
      classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering :: Mathematics',
      ],
      author_email='yoram.zarai@gmail.com',
      license='MIT',
      url='https://github.com/yoramzarai/TpTnOsc',
      packages=['TpTnOsc'],
      install_requires=['numpy',
                        'scipy',
                        'networkx',
                        'matplotlib'],
      python_requires='>=3.8')
