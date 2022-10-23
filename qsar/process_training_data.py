import pandas as pd
import numpy as np
from rdkit.Chem import MACCSkeys
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import rdMolDescriptors



def maccs(mol):

    try:
        maccs_fp = MACCSkeys.GenMACCSKeys(mol)
        return maccs_fp
    except:
        return None

def rdMolDes(mol):
    MDlist=[]
    try:
        MDlist.append(rdMolDescriptors.CalcTPSA(mol))
        MDlist.append(rdMolDescriptors.CalcFractionCSP3(mol))
        MDlist.append(rdMolDescriptors.CalcNumAliphaticCarbocycles(mol))
        MDlist.append(rdMolDescriptors.CalcNumAliphaticHeterocycles(mol))
        MDlist.append(rdMolDescriptors.CalcNumAliphaticRings(mol))
        MDlist.append(rdMolDescriptors.CalcNumAmideBonds(mol))
        MDlist.append(rdMolDescriptors.CalcNumAromaticCarbocycles(mol))
        MDlist.append(rdMolDescriptors.CalcNumAromaticHeterocycles(mol))
        MDlist.append(rdMolDescriptors.CalcNumAromaticRings(mol))
        MDlist.append(rdMolDescriptors.CalcNumHBA(mol))
        MDlist.append(rdMolDescriptors.CalcNumHBD(mol))
        MDlist.append(rdMolDescriptors.CalcNumLipinskiHBA(mol))
        MDlist.append(rdMolDescriptors.CalcNumLipinskiHBD(mol))
        MDlist.append(rdMolDescriptors.CalcNumHeteroatoms(mol))
        MDlist.append(rdMolDescriptors.CalcNumRings(mol))
        MDlist.append(rdMolDescriptors.CalcNumRotatableBonds(mol))
        MDlist.append(rdMolDescriptors.CalcNumSaturatedCarbocycles(mol))
        MDlist.append(rdMolDescriptors.CalcNumSaturatedHeterocycles(mol))
        MDlist.append(rdMolDescriptors.CalcNumSaturatedRings(mol))
        MDlist.append(rdMolDescriptors.CalcHallKierAlpha(mol))
        MDlist.append(rdMolDescriptors.CalcKappa1(mol))
        MDlist.append(rdMolDescriptors.CalcKappa2(mol))
        MDlist.append(rdMolDescriptors.CalcKappa3(mol))
        MDlist.append(rdMolDescriptors.CalcChi0n(mol))
        MDlist.append(rdMolDescriptors.CalcChi0v(mol))
        MDlist.append(rdMolDescriptors.CalcChi1n(mol))
        MDlist.append(rdMolDescriptors.CalcChi1v(mol))
        MDlist.append(rdMolDescriptors.CalcChi2n(mol))
        MDlist.append(rdMolDescriptors.CalcChi2v(mol))
        MDlist.append(rdMolDescriptors.CalcChi3n(mol))
        MDlist.append(rdMolDescriptors.CalcChi3v(mol))
        MDlist.append(rdMolDescriptors.CalcChi4n(mol))
        MDlist.append(rdMolDescriptors.CalcChi4v(mol))
        MDlist.append(rdMolDescriptors.CalcAsphericity(mol))
        MDlist.append(rdMolDescriptors.CalcEccentricity(mol))
        MDlist.append(rdMolDescriptors.CalcInertialShapeFactor(mol))
        MDlist.append(rdMolDescriptors.CalcExactMolWt(mol))
        MDlist.append(rdMolDescriptors.CalcPBF(
            mol))  # Returns the PBF (plane of best fit) descriptor (http://dx.doi.org/10.1021/ci300293f)
        MDlist.append(rdMolDescriptors.CalcPMI1(mol))
        MDlist.append(rdMolDescriptors.CalcPMI2(mol))
        MDlist.append(rdMolDescriptors.CalcPMI3(mol))
        MDlist.append(rdMolDescriptors.CalcRadiusOfGyration(mol))
        MDlist.append(rdMolDescriptors.CalcSpherocityIndex(mol))
        # MDlist.append(rdMolDescriptors.CalcNumBridgeheadAtoms(mol))
        # MDlist.append(rdMolDescriptors.CalcNumAtomStereoCenters(mol))
        # MDlist.append(rdMolDescriptors.CalcNumHeterocycles(mol))
        # MDlist.append(rdMolDescriptors.CalcNumSpiroAtoms(mol))
        # MDlist.append(rdMolDescriptors.CalcNumUnspecifiedAtomStereoCenters(mol))
        MDlist.append(rdMolDescriptors.CalcLabuteASA(mol))
        MDlist.append(rdMolDescriptors.CalcNPR1(mol))
        MDlist.append(rdMolDescriptors.CalcNPR2(mol))
        # for d in rdMolDescriptors.CalcGETAWAY(mol): #197 descr (http://www.vcclab.org/lab/indexhlp/getades.html)
        #    MDlist.append(d)
        for d in rdMolDescriptors.PEOE_VSA_(mol):  # 14 descr
            MDlist.append(d)
        for d in rdMolDescriptors.SMR_VSA_(mol):  # 10 descr
            MDlist.append(d)
        for d in rdMolDescriptors.SlogP_VSA_(mol):  # 12 descr
            MDlist.append(d)
        for d in rdMolDescriptors.MQNs_(mol):  # 42 descr
            MDlist.append(d)
        for d in rdMolDescriptors.CalcCrippenDescriptors(mol):  # 2 descr
            MDlist.append(d)
        for d in rdMolDescriptors.CalcAUTOCORR2D(mol):  # 192 descr
            MDlist.append(d)
        return MDlist
    except:
        return None


def get_fp(name):
    """ Function to get fingerprint from a list of SMILES"""
    fingerprints = []
    Activity=[]

    sdFile = Chem.SDMolSupplier("sdf/{}.sdf".format(name))

    for mol in sdFile:

        try:

            fprint = Chem.GetMorganFingerprintAsBitVect(mol, 3, nBits = 2048,useFeatures=True)

            rd = rdMolDes(mol)
            if rd is None:
                continue
            ma = maccs(mol)
            if ma is None:
                continue
            feature = [x for x in fprint.ToBitString()] + rd  + [x for x in ma.ToBitString()]
            fingerprints.append(feature)

            Activity.append(float(mol.GetProp('pXC50')))
        except:
            print('error')


    return fingerprints, Activity


def clean(name):
    mergedSDF_OUT = Chem.SDWriter('./sdf/{}.sdf'.format(name))
    df = pd.read_csv('../data/{}.csv'.format(name))
    df = df[['Activity_Flag', 'SMILES', 'pXC50']]
    df = df.dropna()
    for index, row in df.iterrows():
        try:
            mol = Chem.MolFromSmiles(row['SMILES'])
            mol.SetProp("pXC50", str(row['pXC50']))
            mergedSDF_OUT.write(mol)
        except:
            print('error1')
    mergedSDF_OUT.close()


names =[
    'EGFR','HTR1A','S1PR1',
]
for name in names:
    clean(name)
    fingers, activities = get_fp(name)
    res = pd.DataFrame(fingers, columns=list(range(2533))).astype('float')
    fp_array = res.to_numpy()
    label = np.array(activities)
    np.save('npy/{}_X.npy'.format(name),fp_array)
    np.save('npy/{}_y.npy'.format(name),label)
