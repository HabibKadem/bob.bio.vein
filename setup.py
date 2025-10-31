#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

from setuptools import dist, setup

dist.Distribution(dict(setup_requires=["bob.extension"]))

from bob.extension.utils import find_packages, load_requirements

install_requires = load_requirements()

setup(
    name="bob.bio.vein",
    version=open("version.txt").read().rstrip(),
    description="Vein Recognition Library",
    url="https://gitlab.idiap.ch/bob/bob.bio.vein",
    license="GPLv3",
    author="Andre Anjos,Pedro Tome",
    author_email="andre.anjos@idiap.ch,pedro.tome@idiap.ch",
    keywords="bob, biometric recognition, evaluation, vein",
    long_description=open("README.rst").read(),
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    install_requires=install_requires,
    entry_points={
        "bob.bio.config": [
            # protocols (must be specified before the database in the cmd)
            # utfvp
            "nom = bob.bio.vein.config.database.utfvp_nom",
            "full = bob.bio.vein.config.database.utfvp_full",
            "1vsall = bob.bio.vein.config.database.utfvp_1vsall",
            # legacy baselines
            "mc = bob.bio.vein.config.maximum_curvature",
            "rlt = bob.bio.vein.config.repeated_line_tracking",
            "wld = bob.bio.vein.config.wide_line_detector",
            # verafinger contactless
            "vera_nom = bob.bio.vein.config.database.verafinger_contactless_nom",
            # dorsal hand veins
            "dorsalhandveins = bob.bio.vein.config.database.dorsalhandveins",
            # cnn+vit model
            "cnn_vit = bob.bio.vein.config.cnn_vit",
        ],
        "bob.bio.database": [
            "utfvp = bob.bio.vein.config.database.utfvp_nom:database",
            "verafinger_contactless = bob.bio.vein.config.database.verafinger_contactless_nom:database",
            "dorsalhandveins = bob.bio.vein.config.database.dorsalhandveins:database",
        ],
        "bob.bio.pipeline": [
            "wld = bob.bio.vein.config.wide_line_detector:pipeline",
            "mc = bob.bio.vein.config.maximum_curvature:pipeline",
            "rlt = bob.bio.vein.config.repeated_line_tracking:pipeline",
            "cnn_vit = bob.bio.vein.config.cnn_vit:pipeline",
        ],
        "console_scripts": [
            "bob_bio_vein_compare_rois.py = bob.bio.vein.script.compare_rois:main",
            "bob_bio_vein_view_sample.py = bob.bio.vein.script.view_sample:main",
            "bob_bio_vein_blame.py = bob.bio.vein.script.blame:main",
            "bob_bio_vein_train_cnn_vit.py = bob.bio.vein.script.train_cnn_vit:main",
            "bob_bio_vein_cnn_vit_examples.py = bob.bio.vein.script.cnn_vit_examples:main",
        ],
    },
    classifiers=[
        "Framework :: Bob",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Natural Language :: English",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
