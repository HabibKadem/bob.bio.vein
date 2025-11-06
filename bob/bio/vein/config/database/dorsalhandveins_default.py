#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

"""Configuration for dorsal hand veins database with default protocol"""

from bob.bio.vein.database.dorsalhandveins import DorsalHandVeinsDatabase

database = DorsalHandVeinsDatabase(protocol="default")
