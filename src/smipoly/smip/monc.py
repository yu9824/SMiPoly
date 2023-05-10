#!/usr/bin/env python
# coding: utf-8

# Copyright (c) 2021 Mitsuru Ohno
#  Use of this source code is governed by a BSD-3-style
#  license that can be found in the LICENSE file.

# 07/27/2021, M. Ohno
# smilesで与えた化合物リストを、モノマー別に分類する
# monomer categolization system of the compound list in SMILES
#
# Refernce:
# https://future-chem.com/rdkit-chemical-rxn/
# https://www.daylight.com/dayhtml_tutorials/languages/smarts/smarts_examples.html
# https://www.daylight.com/dayhtml/doc/theory/theory.smarts.html

from typing import Dict, List
import os
import json
from pathlib import Path

import pandas as pd

from .funclib import genmol, gencSMI, monomer_sel_MFG, monomer_sel_PFG


def moncls(
    df: pd.DataFrame,
    smiColn: str,
    minFG: int = 2,
    maxFG: int = 4,
    dsp_rsl: bool = False,
) -> pd.DataFrame:
    """classify monomers

    Parameters
    ----------
    df : pd.DataFrame
        name of the object
    smiColn : str
        The column label of the SMILES column
    minFG : int, optional
        minimum number of the polymerizable functional groups in the monomer
         for successive polymerization (2 or more), by default 2
    maxFG : int, optional
        maxmum number of the polymerizable functional groups in the monomer
          for successive polymerization (4 or less), by default 4
    dsp_rsl : bool, optional
        display classified result, by default False

    Returns
    -------
    pd.DataFrame
    """
    # The default number of the samle class of FG were limited 2 to 4
    # in the same molecule for poly functionalized monomer.
    # (dataframe, smiColn, minFG = 2, maxFG = 4)

    # read source file
    DF01 = df
    smiColn = smiColn

    # append CO and HCHO for carbonate
    DF02 = DF01.copy()
    if smiColn in DF02.columns.to_list():
        # DF02 = DF02[['CID', smiColn]] #remove this row if not required
        DFadd = pd.DataFrame(
            [
                ["[C-]#[O+]"],
                ["C=O"],
            ],
            columns=[smiColn],
        )
        DF02 = pd.concat([DF02, DFadd], ignore_index=True)
    else:
        print("invalid SMILES column name")

    # drop NA of smiles, and add chemical structure
    DF02["ROMol"] = DF02[smiColn].apply(genmol)
    DF02["smip_cand_mons"] = DF02["ROMol"].apply(gencSMI)

    db_file = os.path.join(
        str(Path(__file__).resolve().parent.parent), "rules"
    )
    with open(os.path.join(db_file, "mon_dic.json"), "r") as f:
        # mon_dic = json.load(f)
        pass
    with open(os.path.join(db_file, "mon_dic_inv.json"), "r") as f:
        mon_dic_inv: Dict[str, str] = json.load(f)
    with open(os.path.join(db_file, "mon_lst.json"), "r") as f:
        monL: Dict[str, List[str]] = json.load(f)
    with open(os.path.join(db_file, "excl_lst.json"), "r") as f:
        exclL: Dict[str, List[str]] = json.load(f)

    monL = {int(k): v for k, v in monL.items()}
    exclL = {int(k): v for k, v in exclL.items()}
    mon_dic_inv = {int(k): v for k, v in mon_dic_inv.items()}

    # classification for mono-functionalized monomer
    # count functional groupe,
    # remove exclude compounds andjudge targetted monomer or not.
    for i in range(
        1, 14
    ):  # fix the range if the monomer defination was modified
        mons = monL[i]
        excls = list(exclL[i])
        DF02[mon_dic_inv[i]] = DF02["ROMol"].apply(
            monomer_sel_MFG, mons=mons, excls=excls
        )
        if dsp_rsl:
            print(i)
            print(
                mon_dic_inv[i],
                " = ",
                len(DF02[DF02[mon_dic_inv[i]]]),
                " / ",
                len(DF02),
            )
        else:
            pass

    # classification for poly-functionalized monomer
    for i in range(
        51, 59
    ):  # fix the range if the monomer defination was modified
        mons = monL[i]
        excls = exclL[i]
        # chk = []
        DF02[mon_dic_inv[i]] = DF02["ROMol"].apply(
            monomer_sel_PFG, mons=mons, excls=excls, minFG=minFG, maxFG=maxFG
        )
        if dsp_rsl:
            print(i)
            print(
                mon_dic_inv[i],
                " = ",
                len(DF02[DF02[mon_dic_inv[i]]]),
                " / ",
                len(DF02),
            )
        else:
            pass
    return DF02


# end
