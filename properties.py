from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Lipinski
from rdkit.Chem import Descriptors
from rdkit.Chem import rdPartialCharges
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Contrib.SA_Score import sascorer
from rdkit.Chem.Draw import SimilarityMaps
from rdkit.Chem import QED
from rdkit.Chem import Draw
import pandas as pd
import numpy as np
import os,sys,shutil
import json,math

def tice_rule_alerts(d):
    # this model is from https://onlinelibrary.wiley.com/doi/10.1002/1526-4998(200101)57:1%3C3::AID-PS269%3E3.0.CO;2-6
    # If you use this function, please consider citing the original authors' paper:
    # https://doi.org/10.1002/1526-4998(200101)57:1<3::AID-PS269>3.0.CO;2-6
    alerts_herbicides = 0
    alerts_insecticides = 0
    if float(d['Molecular_weight']) < 150.0 or float(d['Molecular_weight']) > 500.0:
        alerts_herbicides += 1
        alerts_insecticides += 1

    if float(d['Alogp']) > 5.0:
        alerts_herbicides += 1
    if float(d['Alogp']) < 0.0 or float(d['Alogp']) > 6.5:
        alerts_insecticides += 1

    if float(d['Hydrogen_bond_donors']) > 3:
        alerts_herbicides += 1
    if float(d['Hydrogen_bond_donors']) > 2:
        alerts_insecticides += 1

    if float(d['Hydrogen_bond_acceptors']) < 2 or float(d['Hydrogen_bond_acceptors']) > 12:
        alerts_herbicides += 1
    if float(d['Hydrogen_bond_acceptors']) < 1 or float(d['Hydrogen_bond_acceptors']) > 8:
        alerts_insecticides += 1

    if float(d['Rotatable_bonds']) > 12:
        alerts_herbicides += 1
        alerts_insecticides += 1
    
    d['alerts_herbicides'] = alerts_herbicides
    d['alerts_insecticides'] = alerts_insecticides
    return d


def fungicide(properties_dictionary):
    """
--------------------------------------------------------
'fungicide' is used to score fungicide-likeness by using 3 methods
    usage: fungicide(properties_dictionary)
    input : properties_dictionary - a dictionary describe the properties
    output: threescore dictionary
--------------------------------------------------------
"""
    #######1.BYS#####################################
    arR = properties_dictionary['nAroma_Rings']
    mw = properties_dictionary['Molecular_weight']
    LogP = properties_dictionary['Alogp']
    HBA = properties_dictionary['Hydrogen_bond_acceptors']
    HBD = properties_dictionary['Hydrogen_bond_donors']
    RB = properties_dictionary['Rotatable_bonds']
    arR = float(arR)
    mw = float(mw)
    LogP = float(LogP)
    HBA = float(HBA)
    HBD = float(HBD)
    RB = float(RB)
    if arR == 0:
        s1 = 2.1709
    elif arR == 1:
        s1 = 1.8944
    elif arR == 2:
        s1 = 1.1881
    elif arR == 3:
        s1 = 0.3198
    elif arR == 4:
        s1 = 0.0876
    elif arR == 6:
        s1 = 0.3714
    else:
        s1 = 0
    s1 = float(s1)
    if 30 <= mw < 80:
        s2 = 100
    elif 80 <= mw < 130:
        s2 = 6.88
    elif 130 <= mw < 180:
        s2 = 4.7
    elif 180 <= mw < 230:
        s2 = 2.4605
    elif 230 <= mw < 280:
        s2 = 1.8341
    elif 280 <= mw < 330:
        s2 = 2.0397
    elif 330 <= mw < 380:
        s2 = 1.1212
    elif 380 <= mw < 430:
        s2 = 0.7371
    elif 430 <= mw < 480:
        s2 = 0.2072
    elif 480 <= mw < 530:
        s2 = 0.1472
    elif 530 <= mw < 580:
        s2 = 0.0481
    elif 580 <= mw < 630:
        s2 = 0.1293
    elif 630 <= mw < 680:
        s2 = 0.1
    elif 980 <= mw < 1030:
        s2 = 0.65
    elif 1130 <= mw < 1180:
        s2 = 2.6
    else:
        s2 = 0
    s2 = float(s2)

    if -9 <= LogP < -8:
        s3 = 0.5
    elif -8 <= LogP < -7:
        s3 = 0.5
    elif -7 <= LogP < -6:
        s3 = 0.5
    elif -6 <= LogP < -5:
        s3 = 0.0909
    elif -5 <= LogP < -4:
        s3 = 0.35
    elif -4 <= LogP < -3:
        s3 = 0.1714
    elif -3 <= LogP < -2:
        s3 = 0.05
    elif -2 <= LogP < -1:
        s3 = 0.1111
    elif -1 <= LogP < 0:
        s3 = 0.2486
    elif 0 <= LogP < 1:
        s3 = 0.4341
    elif 1 <= LogP < 2:
        s3 = 0.6056
    elif 2 <= LogP < 3:
        s3 = 0.7128
    elif 3 <= LogP < 4:
        s3 = 1.05
    elif 4 <= LogP < 5:
        s3 = 0.6667
    elif 5 <= LogP < 6:
        s3 = 0.5
    elif 6 <= LogP < 7:
        s3 = 1
    else:
        s3 = 0
    s3 = float(s3)

    if HBA == 0:
        s4 = 4.6
    elif HBA == 1:
        s4 = 3.3348
    elif HBA == 2:
        s4 = 1.6359
    elif HBA == 3:
        s4 = 1.1547
    elif HBA == 4:
        s4 = 1.1973
    elif HBA == 5:
        s4 = 0.4439
    elif HBA == 6:
        s4 = 0.3510
    elif HBA == 7:
        s4 = 0.1234
    elif HBA == 8:
        s4 = 0.065
    elif HBA == 9:
        s4 = 0.1963
    elif HBA == 11:
        s4 = 0.3533
    elif HBA == 13:
        s4 = 0.325
    elif HBA == 16:
        s4 = 1.325
    elif HBA == 18:
        s4 = 1.06
    else:
        s4 = 0
    s4 = float(s4)

    if HBD == 0:
        s5 = 1.8150
    elif HBD == 1:
        s5 = 1.4685
    elif HBD == 2:
        s5 = 0.8204
    elif HBD == 3:
        s5 = 0.2281
    elif HBD == 4:
        s5 = 0.1886
    elif HBD == 5:
        s5 = 0.0634
    elif HBD == 7:
        s5 = 0.6583
    elif HBD == 8:
        s5 = 0.1857
    elif HBD == 10:
        s5 = 0.65
    elif HBD == 11:
        s5 = 1.3
    elif HBD == 12:
        s5 = 0.65
    elif HBD == 14:
        s5 = 2.6
    elif HBD == 15:
        s5 = 0.8667
    else:
        s5 = 0
    s5 = float(s5)

    if RB == 0:
        s6 = 3.0353
    elif RB == 1:
        s6 = 2.2824
    elif RB == 2:
        s6 = 1.2783
    elif RB == 3:
        s6 = 2.5652
    elif RB == 4:
        s6 = 0.9667
    elif RB == 5:
        s6 = 1.0691
    elif RB == 6:
        s6 = 1.0579
    elif RB == 7:
        s6 = 0.9426
    elif RB == 8:
        s6 = 0.5716
    elif RB == 9:
        s6 = 0.4157
    elif RB == 10:
        s6 = 0.2933
    elif RB == 11:
        s6 = 0.9815
    elif RB == 12:
        s6 = 0.7933
    elif RB == 16:
        s6 = 0.1444
    elif RB == 19:
        s6 = 1.3
    elif RB == 20:
        s6 = 1.325
    else:
        s6 = 0
    s6 = float(s6)

    if s1 > 0 and s2 > 0 and s3 > 0 and s4 > 0 and s5 > 0 and s6 > 0:
        D = math.log(s1) + math.log(s2) + math.log(3) + math.log(s4) + math.log(s5) + math.log(s6)
        RDL = math.exp(float(D) / 6)
    else:
        RDL = 0

    #######2.QED####################
    m = (-float(mw) + float('320.637')) / float('-75.7229')
    o = (-float(HBA) + float('1.795')) / float('1.203')
    p = (-float(HBD) + float('-0.0003')) / float('0.5961')
    q = (-float(RB) + float('1.411')) / float('3.33')
    r = (-float(arR) + float('-0.05625')) / float('-1.719')
    g = (float('0.9953') + float('56.3117') * math.e ** ((-math.e ** float(m)) + float(m) + float('1.0'))) / float(
        '57.307')
    i = (float('1.319') + float('166') * math.e ** ((-math.e ** float(o)) + float(o) + float('1.0'))) / float('167.319')
    j = (float('1.101') + float('384.2301') * math.e ** ((-math.e ** float(p)) + float(p) + float('1.0'))) / float(
        '385.3311')
    k = (float('-3.253') + float('116') * math.e ** ((-math.e ** float(q)) + float(q) + float('1.0'))) / float(
        '112.747')
    l = (float('1.005') + float('223.7') * math.e ** ((-math.e ** float(r)) + float(r) + float('1.0'))) / float(
        '224.584')
    n = (-float(LogP) + float('0.7263')) / float('1.173')
    h = (float('0.8808') + float('63.79') * math.e ** ((-math.e ** float(n)) + float(n) + float('1.0'))) / float(
        '64.6889')
    if g > 0 and h > 0 and i > 0 and j > 0 and k > 0 and l > 0:
        QEX = math.e ** (1.0 / float('6.0') * (
                math.log(g) + math.log(h) + math.log(i) + math.log(j) + math.log(k) + math.log(l)))
    else:
        QEX = 0

    ####3.GAU#######################
    nRing = properties_dictionary['nRings']
    nN = properties_dictionary['nN']
    nO = properties_dictionary['nO']
    PSA = properties_dictionary['Polar_surface_area']
    j = math.e ** (-((float(mw) - float('290.8')) / float('121.3')) ** float('2.0'))
    l = math.e ** (-((float(HBA) - float('0.8737')) / float('3.48')) ** float('2.0'))
    m = math.e ** (-((float(HBD) - float('-9.41')) / float('3.937')) ** float('2.0'))
    n = math.e ** (-((float(RB) - float('0.7218')) / float('6.752')) ** float('2.0'))
    o = math.e ** (-((float(nRing) - float('1.102')) / float('1.46')) ** float('2.0'))
    q = math.e ** (-((float(nN) - float('1.683')) / float('2.266')) ** float('2.0'))
    r = math.e ** (-((float(nO) - float('-0.6667')) / float('4.129')) ** float('2.0'))
    k = math.e ** (-((float(LogP) - float('1.097')) / float('1.841')) ** float('2.0'))
    p = math.e ** (-((float(PSA) - float('36.83')) / float('85.5')) ** float('2.0'))
    KLS = j + k + l + m + n + o + p + q + r

    properties_dictionary['fungicide_RDL'] = RDL
    properties_dictionary['fungicide_QEI'] = QEX
    properties_dictionary['fungicide_Gau'] = KLS
    ################################
    return properties_dictionary


def herbicide(properties_dictionary):
    """
--------------------------------------------------------
'herbicide' is used to score herbicide-likeness by using 3 methods
    usage: fungicide(properties_dictionary)
    input : properties_dictionary - a dictionary describe the properties
    output: threescore dictionary
--------------------------------------------------------
"""
    ####1.BYS#############################
    arR = properties_dictionary['nAroma_Rings']
    mw = properties_dictionary['Molecular_weight']
    LogP = properties_dictionary['Alogp']
    HBA = properties_dictionary['Hydrogen_bond_acceptors']
    HBD = properties_dictionary['Hydrogen_bond_donors']
    RB = properties_dictionary['Rotatable_bonds']
    arR = float(arR)
    mw = float(mw)
    LogP = float(LogP)
    HBA = float(HBA)
    HBD = float(HBD)
    RB = float(RB)
    if arR == 0:
        s1 = 2.2179
    elif arR == 1:
        s1 = 2.631
    elif arR == 2:
        s1 = 0.9286
    elif arR == 3:
        s1 = 0.2323
    elif arR == 4:
        s1 = 0.0159
    elif arR == 5:
        s1 = 0.0595
    else:
        s1 = 0
    s1 = float(s1)
    if 20 <= mw < 70:
        s2 = 14.7
    elif 70 <= mw < 120:
        s2 = 31.3653
    elif 120 <= mw < 170:
        s2 = 3.2288
    elif 170 <= mw < 220:
        s2 = 4.6125
    elif 220 <= mw < 270:
        s2 = 3.1664
    elif 270 <= mw < 320:
        s2 = 1.1043
    elif 320 <= mw < 370:
        s2 = 0.95326
    elif 370 <= mw < 420:
        s2 = 0.6082
    elif 420 <= mw < 470:
        s2 = 0.5253
    elif 470 <= mw < 520:
        s2 = 0.2838
    elif 570 <= mw < 620:
        s2 = 0.1845
    elif 620 <= mw < 670:
        s2 = 0.1153
    else:
        s2 = 0
    s2 = float(s2)

    if -5 <= LogP < -4:
        s3 = 0.123
    elif -4 <= LogP < -3:
        s3 = 0.240655
    elif -3 <= LogP < -2:
        s3 = 0.297584
    elif -2 <= LogP < -1:
        s3 = 0.7136
    elif -1 <= LogP < 0:
        s3 = 0.888
    elif 0 <= LogP < 1:
        s3 = 1.00488
    elif 1 <= LogP < 2:
        s3 = 1.6291
    elif 2 <= LogP < 3:
        s3 = 1.5841
    elif 3 <= LogP < 4:
        s3 = 1.001581
    elif 4 <= LogP < 5:
        s3 = 0.7096
    else:
        s3 = 0
    s3 = float(s3)

    if HBA == 0:
        s4 = 4.193224
    elif HBA == 1:
        s4 = 1.719932
    elif HBA == 2:
        s4 = 1.817685
    elif HBA == 3:
        s4 = 1.115121
    elif HBA == 4:
        s4 = 0.531446
    elif HBA == 5:
        s4 = 0.662036
    elif HBA == 6:
        s4 = 0.435146
    elif HBA == 7:
        s4 = 1.059177
    elif HBA == 8:
        s4 = 0.9884
    elif HBA == 9:
        s4 = 1.1275
    elif HBA == 10:
        s4 = 2.180476
    elif HBA == 11:
        s4 = 1.38377
    elif HBA == 12:
        s4 = 0.79072
    elif HBA == 13:
        s4 = 0.9225
    else:
        s4 = 0
    s4 = float(s4)

    if HBD == 0:
        s5 = 1.82769
    elif HBD == 1:
        s5 = 0.959184
    elif HBD == 2:
        s5 = 0.9034
    elif HBD == 3:
        s5 = 0.58084
    elif HBD == 4:
        s5 = 0.28076
    elif HBD == 5:
        s5 = 0.1845
    else:
        s5 = 0
    s5 = float(s5)

    if RB == 0:
        s6 = 5.166052
    elif RB == 1:
        s6 = 1.845018
    elif RB == 2:
        s6 = 1.19926
    elif RB == 3:
        s6 = 1.664528
    elif RB == 4:
        s6 = 1.077621
    elif RB == 5:
        s6 = 1.291513
    elif RB == 6:
        s6 = 0.802182
    elif RB == 7:
        s6 = 0.99048
    elif RB == 8:
        s6 = 0.855009
    elif RB == 9:
        s6 = 0.658935
    elif RB == 10:
        s6 = 0.601636
    elif RB == 11:
        s6 = 0.349058
    elif RB == 12:
        s6 = 0.123
    elif RB == 13:
        s6 = 0.307503
    elif RB == 14:
        s6 = 0.15375
    elif RB == 15:
        s6 = 0.230627
    else:
        s6 = 0
    s6 = float(s6)

    if s1 > 0 and s2 > 0 and s3 > 0 and s4 > 0 and s5 > 0 and s6 > 0:
        D = math.log(s1) + math.log(s2) + math.log(3) + math.log(s4) + math.log(s5) + math.log(s6)
        RDL = math.exp(float(D) / 6)
    else:
        RDL = 0

    ######2.QED#################
    m = (-float(mw) + float('263.4605')) / float('83.8494')
    o = (-float(HBA) + float('1.63')) / float('1.351')
    p = (-float(HBD) + float('4.4773')) / float('2.4865')
    q = (-float(RB) + float('4.937')) / float('-3.014')
    r = (-float(arR) + float('0.9019')) / float('0.90075')
    g = (float('0.2064') + float('11.0583') * math.e ** ((-math.e ** float(m)) + float(m) + float('1.0'))) / float(
        '11.2647')
    i = (float('9.125') + float('118.7') * math.e ** ((-math.e ** float(o)) + float(o) + float('1.0'))) / float(
        '117.8908')
    j = (float('221.5') + float('-218.9781') * math.e ** ((-math.e ** float(p)) + float(p) + float('1.0'))) / float(
        '213.0339')
    k = (float('6.399') + float('136.1') * math.e ** ((-math.e ** float(q)) + float(q) + float('1.0'))) / float(
        '142.499')
    l = (float('-15.8891') + float('262.1093') * math.e ** ((-math.e ** float(r)) + float(r) + float('1.0'))) / float(
        '246.2202')
    n = (-float(LogP) + float('1.371')) / float('-1.293')
    h = (float('-0.8092') + float('68.39') * math.e ** ((-math.e ** float(n)) + float(n) + float('1.0'))) / float(
        '67.5808')
    if g > 0 and h > 0 and i > 0 and j > 0 and k > 0 and l > 0:
        QEX = math.e ** (float('1.0') / float('6.0') * (
                math.log(g) + math.log(h) + math.log(i) + math.log(j) + math.log(k) + math.log(l)))
    else:
        QEX = 0

    #####3.GAU#################
    nRing = properties_dictionary['nRings']
    nN = properties_dictionary['nN']
    nO = properties_dictionary['nO']
    PSA = properties_dictionary['Polar_surface_area']
    j = math.e ** (-((float(mw) - float('288.6')) / float('141.6')) ** float('2.0'))
    l = math.e ** (-((float(HBA) - float('2.714')) / float('3.294')) ** float('2.0'))
    m = math.e ** (-((float(HBD) - float('1.044')) / float('1.267')) ** float('2.0'))
    n = math.e ** (-((float(PSA) - float('54.49')) / float('66.42')) ** float('2.0'))
    o = math.e ** (-((float(nRing) - float('1.637')) / float('1.13')) ** float('2.0'))
    p = math.e ** (-((float(nN) - float('1.071')) / float('0.9395')) ** float('2.0'))
    q = math.e ** (-((float(nO) - float('1.966')) / float('3.214')) ** float('2.0'))
    r = math.e ** (-((float(RB) - float('4.456')) / float('4.15')) ** float('2.0'))
    k = math.e ** (-((float(LogP) - float('1.006')) / float('1.928')) ** float('2.0'))
    KLS = j + k + l + m + n + o + p + q + r

    ############################
    properties_dictionary['herbicide_RDL'] = RDL
    properties_dictionary['herbicide_QEI'] = QEX
    properties_dictionary['herbicide_Gau'] = KLS
    ################################
    return properties_dictionary


def insecticide(properties_dictionary):
    """
--------------------------------------------------------
'insecticide' is used to score insecticide-likeness by using 3 methods
    usage: fungicide(properties_dictionary)
    input : properties_dictionary - a dictionary describe the properties
    output: threescore dictionary
--------------------------------------------------------
"""
    ######1.BYS###############
    arR = properties_dictionary['nAroma_Rings']
    mw = properties_dictionary['Molecular_weight']
    LogP = properties_dictionary['Alogp']
    HBA = properties_dictionary['Hydrogen_bond_acceptors']
    HBD = properties_dictionary['Hydrogen_bond_donors']
    RB = properties_dictionary['Rotatable_bonds']
    arR = float(arR)
    mw = float(mw)
    LogP = float(LogP)
    HBA = float(HBA)
    HBD = float(HBD)
    RB = float(RB)
    if arR == 0:
        s1 = 4.5085
    elif arR == 1:
        s1 = 1.9302
    elif arR == 2:
        s1 = 0.586
    elif arR == 3:
        s1 = 0.232
    elif arR == 4:
        s1 = 0.0845
    else:
        s1 = 0
    s1 = float(s1)
    if 20 <= mw < 70:
        s2 = 25.4
    elif 70 <= mw < 120:
        s2 = 23.4
    elif 120 <= mw < 170:
        s2 = 4.8833
    elif 170 <= mw < 220:
        s2 = 2.7333
    elif 220 <= mw < 270:
        s2 = 1.9
    elif 270 <= mw < 320:
        s2 = 1.4971
    elif 320 <= mw < 370:
        s2 = 0.9767
    elif 370 <= mw < 420:
        s2 = 0.7406
    elif 420 <= mw < 470:
        s2 = 0.2715
    elif 470 <= mw < 520:
        s2 = 0.4725
    elif 520 <= mw < 570:
        s2 = 0.6067
    elif 570 <= mw < 620:
        s2 = 0.1967
    elif 620 <= mw < 670:
        s2 = 0.975
    elif 670 <= mw < 720:
        s2 = 0.4538
    elif 720 <= mw < 770:
        s2 = 1.95
    elif 870 <= mw < 920:
        s2 = 9.8
    else:
        s2 = 0
    s2 = float(s2)

    if -6 <= LogP < -5:
        s3 = 0.2857
    elif -4 <= LogP < -3:
        s3 = 0.3391
    elif -3 <= LogP < -2:
        s3 = 0.1581
    elif -2 <= LogP < -1:
        s3 = 0.1840
    elif -1 <= LogP < 0:
        s3 = 0.2925
    elif 0 <= LogP < 1:
        s3 = 0.6277
    elif 1 <= LogP < 2:
        s3 = 1.1846
    elif 2 <= LogP < 3:
        s3 = 2.0717
    elif 3 <= LogP < 4:
        s3 = 5.8028
    elif 4 <= LogP < 5:
        s3 = 4.5077
    elif 5 <= LogP < 6:
        s3 = 6.85
    else:
        s3 = 0
    s3 = float(s3)

    if HBA == 0:
        s4 = 9.5909
    elif HBA == 1:
        s4 = 1.2915
    elif HBA == 2:
        s4 = 1.7074
    elif HBA == 3:
        s4 = 1.3093
    elif HBA == 4:
        s4 = 0.775
    elif HBA == 5:
        s4 = 0.6665
    elif HBA == 6:
        s4 = 0.4604
    elif HBA == 7:
        s4 = 0.3611
    elif HBA == 9:
        s4 = 0.1111
    elif HBA == 10:
        s4 = 0.7091
    elif HBA == 13:
        s4 = 1
    elif HBA == 14:
        s4 = 0.9833
    elif HBA == 15:
        s4 = 1
    elif HBA == 16:
        s4 = 3.9
    else:
        s4 = 0
    s4 = float(s4)

    if HBD == 0:
        s5 = 2.7141
    elif HBD == 1:
        s5 = 0.8003
    elif HBD == 2:
        s5 = 0.4295
    elif HBD == 3:
        s5 = 0.3435
    elif HBD == 4:
        s5 = 0.1283
    elif HBD == 5:
        s5 = 0.39
    elif HBD == 6:
        s5 = 0.1538
    elif HBD == 9:
        s5 = 0.6667
    else:
        s5 = 0
    s5 = float(s5)

    if RB == 0:
        s6 = 6.64
    elif RB == 1:
        s6 = 1.5129
    elif RB == 2:
        s6 = 0.39
    elif RB == 3:
        s6 = 0.7641
    elif RB == 4:
        s6 = 0.6566
    elif RB == 5:
        s6 = 1.2208
    elif RB == 6:
        s6 = 1.0188
    elif RB == 7:
        s6 = 1.9116
    elif RB == 8:
        s6 = 1
    elif RB == 9:
        s6 = 1.2214
    elif RB == 10:
        s6 = 0.85
    elif RB == 11:
        s6 = 0.2649
    elif RB == 12:
        s6 = 0.6533
    elif RB == 13:
        s6 = 0.2167
    elif RB == 14:
        s6 = 0.1667
    elif RB == 15:
        s6 = 0.25
    else:
        s6 = 0
    s6 = float(s6)

    if s1 > 0 and s2 > 0 and s3 > 0 and s4 > 0 and s5 > 0 and s6 > 0:
        D = math.log(s1) + math.log(s2) + math.log(3) + math.log(s4) + math.log(s5) + math.log(s6)
        RDL = math.exp(float(D) / 6)
    else:
        RDL = 0

    ######2.QED###############
    m = (-float(mw) + float('327.9907')) / float('-79.4646')
    p = (-float(HBA) + float('3.087')) / float('-1.876')
    q = (-float(HBD) + float('-12.96')) / float('-7.109')
    r = (-float(RB) + float('6.4934')) / float('-2.07')
    s = (-float(arR) + float('0.01748')) / float('1.2825')
    f1 = (float('2.2657') + float('38.759') * math.e ** ((-math.e ** float(m)) + float(m) + float('1.0'))) / float(
        '41.0247')
    f3 = (float('1.986') + float('107') * math.e ** ((-math.e ** float(p)) + float(p) + float('1.0'))) / float(
        '108.986')
    f4 = (float('0.6185') + float('8560.21') * math.e ** ((-math.e ** float(q)) + float(q) + float('1.0'))) / float(
        '295.9772')
    f5 = (float('7.6888') + float('71.5256') * math.e ** ((-math.e ** float(r)) + float(r) + float('1.0'))) / float(
        '79.2144')
    f6 = (float('-23.638') + float('240.6037') * math.e ** ((-math.e ** float(s)) + float(s) + float('1.0'))) / float(
        '216.9657')
    n = (-float(LogP) + float('1.754')) / float('1.485')
    f2 = (float('0.8725') + float('59.97') * math.e ** ((-math.e ** float(n)) + float(n) + float('1.0'))) / float(
        '60.8426')
    if f1 and f2 and f3 and f4 and f5 and f6 > 0:
        QEX = math.e ** (float('1.0') / float('6.0') * (
                math.log(f1) + math.log(f2) + math.log(f3) + math.log(f4) + math.log(f5) + math.log(f6)))
    else:
        QEX = 0

    #######3.GAU##############
    nRing = properties_dictionary['nRings']
    nN = properties_dictionary['nN']
    nO = properties_dictionary['nO']
    PSA = properties_dictionary['Polar_surface_area']
    j = math.e ** (-((float(mw) - float('308.6')) / float('134.6')) ** float('2.0'))
    l = math.e ** (-((float(HBA) - float('2.776')) / float('2.511')) ** float('2.0'))
    m = math.e ** (-((float(HBD) - float('-5.302')) / float('3.828')) ** float('2.0'))
    n = math.e ** (-((float(PSA) - float('57.92')) / float('67.62')) ** float('2.0'))
    o = math.e ** (-((float(nRing) - float('1.004')) / float('2.387')) ** float('2.0'))
    p = math.e ** (-((float(nN) - float('-5.502')) / float('5.1')) ** float('2.0'))
    q = math.e ** (-((float(nO) - float('2.78')) / float('1.707')) ** float('2.0'))
    r = math.e ** (-((float(RB) - float('6.002')) / float('3.577')) ** float('2.0'))
    k = math.e ** (-((float(LogP) - float('2.152')) / float('2.24')) ** float('2.0'))
    KLS = j + k + l + m + n + o + p + q + r

    ###########################
    properties_dictionary['insecticide_RDL'] = RDL
    properties_dictionary['insecticide_QEI'] = QEX
    properties_dictionary['insecticide_Gau'] = KLS
    ################################
    return properties_dictionary

# 导入数据
df = pd.read_csv('data.csv')
df.head(5)

# 计算分子性质
i = 1
data = list()
for smi in df['Canonical SMILES']:
    properties = dict()
    try:
        mol = Chem.MolFromSmiles(smi)
        ## 计算N，O原子数量，并存入字典内
        nN = 0
        nO = 0
        for atom in mol.GetAtoms():
            atom_type = atom.GetSymbol()
            if atom_type == "N":
                nN += 1            
            if atom_type == "O":
                nO += 1
        properties['nN'] = nN
        properties['nO'] = nO
        ## 计算芳香键数量
        num_aromatic_bonds = 0
        for bond in mol.GetBonds():
            if bond.GetIsAromatic():
                num_aromatic_bonds += 1
        ## 查看字典可知计算的相应描述符方法
        properties['Molecular_charge'] = Chem.GetFormalCharge(mol)
        properties['Molecular_formula'] = Chem.rdMolDescriptors.CalcMolFormula(mol)
        properties['Molecular_weight'] = Descriptors.MolWt(mol)
        properties['Number_of_atoms'] = mol.GetNumAtoms()
        properties['nAroma_Rings'] = Lipinski.NumAromaticRings(mol)
        properties['nAroma_Bonds'] = num_aromatic_bonds
        properties['QED_Drug-Likeness'] = QED.default(mol)

        properties['Number_of_heavyatoms'] = mol.GetNumHeavyAtoms()
        properties['Alogp'] = Descriptors.MolLogP(mol)
        properties['Hydrogen_bond_acceptors'] = Lipinski.NumHAcceptors(mol)
        properties['Hydrogen_bond_donors'] = Lipinski.NumHDonors(mol)
        properties['Polar_surface_area'] = Descriptors.TPSA(mol)
        properties['Rotatable_bonds'] = Lipinski.NumRotatableBonds(mol)
        properties['Molar_refractivity'] = Descriptors.MolMR(mol)
        properties['nRings'] = len(Chem.GetSymmSSSR(mol))
        properties['Synthetic_Accessibility'] = sascorer.calculateScore(mol)
        # 绘制电荷图，，由于运行完，内存会炸，所以整个代码执行了两次，第一次生成了各原子电荷图，第二次，将生成图片部分注释掉，以防止代码报错，但保留生成的相应文件名，并存入字典中
        image = f'charge{i}.png'
        # AllChem.ComputeGasteigerCharges(mol)
        # contribs = [round(mol.GetAtomWithIdx(i).GetDoubleProp('_GasteigerCharge'), 2) for i in range(mol.GetNumAtoms())]
        # fig = SimilarityMaps.GetSimilarityMapFromWeights(mol, contribs, contourLines=10)
        # fig.savefig(image, bbox_inches='tight')
        properties['charge_image'] = image
        
        ## 计算课题组的农药指标
        properties = tice_rule_alerts(properties)
        properties = fungicide(properties)
        properties = herbicide(properties)
        properties = insecticide(properties)
        data.append(properties)
    # 当无smiles时，同样产生输出，使总数目 以及各条目能对应上
    except:
        properties['nN'] = '-'
        properties['nO'] = '-'
        properties['Molecular_charge'] = '-'
        properties['Molecular_formula'] = '-'
        properties['Molecular_weight'] = '-'
        properties['Number_of_atoms'] = '-'
        properties['nAroma_Rings'] = '-'
        properties['nAroma_Bonds'] = '-'
        properties['QED_Drug-Likeness'] = '-'

        properties['Number_of_heavyatoms'] = '-'
        properties['Alogp'] = '-'
        properties['Hydrogen_bond_acceptors'] = '-'
        properties['Hydrogen_bond_donors'] = '-'
        properties['Polar_surface_area'] = '-'
        properties['Rotatable_bonds'] = '-'
        properties['Molar_refractivity'] = '-'
        properties['nRings'] = '-'
        properties['Synthetic_Accessibility'] = '-'
        properties['charge_image'] = '-'

        properties['alerts_herbicides'] = '-'
        properties['alerts_insecticides'] = '-'
        properties['fungicide_RDL'] = '-'
        properties['fungicide_QEI'] = '-'
        properties['fungicide_Gau'] = '-'
        properties['herbicide_RDL'] = '-'
        properties['herbicide_QEI'] = '-'
        properties['herbicide_Gau'] = '-'
        properties['insecticide_RDL'] = '-'
        properties['insecticide_QEI'] = '-'
        properties['insecticide_Gau'] = '-'
        print(f'molecule {i} has error!')
        data.append(properties)
    i += 1

df_new = pd.concat([df,df1],axis=1)
# 检查结果是否有误
df_new.head(5)

# 剔除不需要的结果列，并查看
df_ = df_new.drop(['nN','nO','Molecular_formula','Molecular_weight','Number_of_atoms','nAroma_Rings','nAroma_Bonds','QED_Drug-Likeness'],axis=1)
df_.head(5)

# 导出csv文件
df_.to_csv('properties.csv',index=None)
