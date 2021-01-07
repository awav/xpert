from setuptools import setup

setup(
    name="xpert",
    version="0.1",
    py_modules=["xpert"],
    install_requires=["click", "termcolor"],
    entry_points="""
        [console_scripts]
        xpert=xpert:main
    """,
)