import setuptools

setuptools.setup(
    name='aspo',
    version='0.1.0',
    description='Auxiliary package for "An Algebraic Perspective to Signomial Optimization"',
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Natural Language :: English',
        'License :: OSI Approved :: Apache Software License',
        'Development Status :: 3 - alpha',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Mathematics'
    ],
    python_requires='>=3.6',
    install_requires=["ecos >= 2",
                      "numpy >= 1.14",
                      "scipy >= 1.1",
                      "sageopt >= 0.5.3",
                      "matplotlib",
                      "sympy"],
)
