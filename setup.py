# -*- coding: utf-8 -*-
u"""rswarp setup script

:copyright: Copyright (c) 2016 RadiaSoft LLC.  All Rights Reserved.
:license: http://www.apache.org/licenses/LICENSE-2.0.html
"""
from pykern.pksetup import setup

setup(
    name='rswarp',
    author='RadiaSoft LLC',
    author_email='pip@radiasoft.net',
    description='Python tools for use with Warp',
    license='http://www.apache.org/licenses/LICENSE-2.0.html',
    url='https://github.com/radiasoft/rswarp',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python',
        'Topic :: Utilities',
    ],
    scripts=['rswarp/run_files/tec/run_warp.py', 'rswarp/run_files/tec/run_warp_nersc.py']
)
