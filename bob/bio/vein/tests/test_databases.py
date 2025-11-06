#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Thu May 24 10:41:42 CEST 2012

import os

from bob.extension.download import get_file


def test_verafinger_contactless():
    from bob.bio.vein.database.verafinger_contactless import (
        VerafingerContactless,
    )

    # Getting the absolute path
    urls = VerafingerContactless.urls()
    filename = get_file("verafinger_contactless.tar.gz", urls)

    # Removing the file before the test
    try:
        os.remove(filename)
    except Exception:
        pass

    # nom
    nom_parameters = {
        "N_dev": 65,
        "N_eval": 68,
        "N_session_references": 2,
        "N_session_probes": 3,
        "N_hands": 2,
    }

    protocols_parameters = {
        "nom": nom_parameters,
    }

    def _check_protocol(p, parameters, eval=False):
        database = VerafingerContactless(protocol=p)

        assert (
            len(database.references(group="dev"))
            == parameters["N_dev"]
            * parameters["N_hands"]
            * parameters["N_session_references"]
        )
        assert (
            len(database.probes(group="dev"))
            == parameters["N_dev"]
            * parameters["N_hands"]
            * parameters["N_session_probes"]
        )

        if eval:
            assert (
                len(database.references(group="eval"))
                == parameters["N_eval"]
                * parameters["N_hands"]
                * parameters["N_session_references"]
            )
            assert (
                len(database.probes(group="eval"))
                == parameters["N_eval"]
                * parameters["N_hands"]
                * parameters["N_session_probes"]
            )

        return p

    checked_protocols = []

    checked_protocols.append(
        _check_protocol("nom", protocols_parameters["nom"], eval=True)
    )

    for p in VerafingerContactless.protocols():
        assert p in checked_protocols, "Protocol {} untested".format(p)


def test_utfvp():
    from bob.bio.vein.database.utfvp import UtfvpDatabase

    # Getting the absolute path
    urls = UtfvpDatabase.urls()
    filename = get_file("utfvp_csv.tar.gz", urls)

    # Removing the file before the test
    try:
        os.remove(filename)
    except Exception:
        pass

    N_SUBJECTS, N_SESSIONS = 60, 4
    # nom
    nom_parameters = {
        "N_train": 10,
        "N_dev": 18,
        "N_eval": 32,
        "N_session": N_SESSIONS // 2,
        "N_session_training": N_SESSIONS,
        "N_fingers": 6,
        "N_fingers_training": 6,
    }
    # full
    full_parameters = {
        "N_train": 0,
        "N_dev": N_SUBJECTS,
        "N_eval": 0,
        "N_session": N_SESSIONS,
        "N_fingers": 6,
    }

    # 1vsall
    onevsall_parameters = {
        "N_train": 35,
        "N_dev": 65,
        "N_eval": 0,
        "N_session": N_SESSIONS,
        "N_session_training": N_SESSIONS,
        "N_fingers": 5,
        "N_fingers_training": 1,
    }
    # subnom
    subnom_parameters = {
        "N_train": 10,
        "N_dev": 18,
        "N_eval": 32,
        "N_session": N_SESSIONS // 2,
        "N_session_training": N_SESSIONS,
        "N_fingers": 1,
        "N_fingers_training": 1,
    }

    # subfull
    subfull_parameters = {
        "N_train": 0,
        "N_dev": N_SUBJECTS,
        "N_eval": 0,
        "N_session": N_SESSIONS,
        "N_fingers": 1,
    }

    protocols_parameters = {
        "nom": nom_parameters,
        "full": full_parameters,
        "1vsall": onevsall_parameters,
        "subnom": subnom_parameters,
        "subfull": subfull_parameters,
    }

    def _check_protocol(p, parameters, train=False, eval=False):
        database = UtfvpDatabase(protocol=p)

        if train:
            assert (
                len(database.background_model_samples())
                == parameters["N_train"]
                * parameters["N_fingers_training"]
                * parameters["N_session_training"]
            )

        assert (
            len(database.references(group="dev"))
            == parameters["N_dev"]
            * parameters["N_fingers"]
            * parameters["N_session"]
        )
        assert (
            len(database.probes(group="dev"))
            == parameters["N_dev"]
            * parameters["N_fingers"]
            * parameters["N_session"]
        )

        if eval:
            assert (
                len(database.references(group="eval"))
                == parameters["N_eval"]
                * parameters["N_fingers"]
                * parameters["N_session"]
            )
            assert (
                len(database.probes(group="eval"))
                == parameters["N_eval"]
                * parameters["N_fingers"]
                * parameters["N_session"]
            )

        return p

    checked_protocols = []

    checked_protocols.append(
        _check_protocol(
            "nom", protocols_parameters["nom"], train=True, eval=True
        )
    )
    checked_protocols.append(
        _check_protocol(
            "full", protocols_parameters["full"], train=False, eval=False
        )
    )
    checked_protocols.append(
        _check_protocol(
            "1vsall", protocols_parameters["1vsall"], train=True, eval=False
        )
    )
    checked_protocols.append(
        _check_protocol(
            "nomLeftIndex",
            protocols_parameters["subnom"],
            train=True,
            eval=True,
        )
    )
    checked_protocols.append(
        _check_protocol(
            "nomLeftMiddle",
            protocols_parameters["subnom"],
            train=True,
            eval=True,
        )
    )
    checked_protocols.append(
        _check_protocol(
            "nomLeftRing", protocols_parameters["subnom"], train=True, eval=True
        )
    )
    checked_protocols.append(
        _check_protocol(
            "nomRightIndex",
            protocols_parameters["subnom"],
            train=True,
            eval=True,
        )
    )
    checked_protocols.append(
        _check_protocol(
            "nomRightMiddle",
            protocols_parameters["subnom"],
            train=True,
            eval=True,
        )
    )
    checked_protocols.append(
        _check_protocol(
            "nomRightRing",
            protocols_parameters["subnom"],
            train=True,
            eval=True,
        )
    )
    checked_protocols.append(
        _check_protocol(
            "fullLeftIndex",
            protocols_parameters["subfull"],
            train=False,
            eval=False,
        )
    )
    checked_protocols.append(
        _check_protocol(
            "fullLeftMiddle",
            protocols_parameters["subfull"],
            train=False,
            eval=False,
        )
    )
    checked_protocols.append(
        _check_protocol(
            "fullLeftRing",
            protocols_parameters["subfull"],
            train=False,
            eval=False,
        )
    )
    checked_protocols.append(
        _check_protocol(
            "fullRightIndex",
            protocols_parameters["subfull"],
            train=False,
            eval=False,
        )
    )
    checked_protocols.append(
        _check_protocol(
            "fullRightMiddle",
            protocols_parameters["subfull"],
            train=False,
            eval=False,
        )
    )
    checked_protocols.append(
        _check_protocol(
            "fullRightRing",
            protocols_parameters["subfull"],
            train=False,
            eval=False,
        )
    )

    for p in UtfvpDatabase.protocols():
        assert p in checked_protocols, "Protocol {} untested".format(p)


def test_dorsalhandveins_basic():
    """Test that DorsalHandVeinsDatabase can be instantiated and has proper structure"""
    from bob.bio.vein.database.dorsalhandveins import DorsalHandVeinsDatabase

    # Test instantiation with default protocol
    database = DorsalHandVeinsDatabase(protocol="default")
    
    # Verify the database has the expected attributes
    assert database.name == "dorsalhandveins"
    assert database.protocol == "default"
    
    # Verify protocols method returns expected list
    protocols = DorsalHandVeinsDatabase.protocols()
    assert isinstance(protocols, list)
    assert "default" in protocols

