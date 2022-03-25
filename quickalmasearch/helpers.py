import numpy as np
import requests
from bs4 import BeautifulSoup
import astropy.units as u
import astropy.constants as ac
import scipy.constants as sc
from astropy.modeling import models
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

ckms = sc.c * 1e-3
ccms = sc.c * 1e2
h = sc.h
k = sc.k

pf_table_path = "https://cdms.astro.uni-koeln.de/classic/entries/partition_function.html"

# relevant columns for intensity estimates
columns = (
    "Species",
    "Resolved QNs",
    "Freq-GHz(rest frame,redshifted)",
    "Meas Freq-GHz(rest frame,redshifted)",
    "Log<sub>10</sub> (A<sub>ij</sub>)",
    "S<sub>ij</sub>&#956;<sup>2</sup> (D<sup>2</sup>)",
    "Upper State Degeneracy",
    "E_U (K)",
)


def select_and_rename_columns(table):
    table = table[columns]
    table.rename_column("Resolved QNs", "QNs")
    table.rename_column("Freq-GHz(rest frame,redshifted)", "nu0 [GHz]")
    table.rename_column("Meas Freq-GHz(rest frame,redshifted)", "Meas nu0 [GHz]")
    table.rename_column("Log<sub>10</sub> (A<sub>ij</sub>)", "logA [s^-1]")
    table.rename_column(
        "S<sub>ij</sub>&#956;<sup>2</sup> (D<sup>2</sup>)", "Smu2 [D^2]"
    )
    table.rename_column("Upper State Degeneracy", "g_u")
    table.rename_column("E_U (K)", "E_u [K]")
    return table


def resolve_freq_duplication(table):
    table["nu0 [GHz]"] = table["nu0 [GHz]"].astype(float)
    if table.mask is not None:
        for i in table.mask["nu0 [GHz]"].nonzero():
            table["nu0 [GHz]"][i] = table["Meas nu0 [GHz]"][i]
    table.remove_column("Meas nu0 [GHz]")
    return table


def resolve_hfs_rot_duplication(table, remove_hfs=False, hfs_indicator="F"):
    hfs_arg = [i for i, qn in enumerate(table["QNs"]) if hfs_indicator in qn]
    rot_arg = [i for i, qn in enumerate(table["QNs"]) if not hfs_indicator in qn]
    if remove_hfs:
        table.remove_rows(hfs_arg)
    else:
        table.remove_rows(rot_arg)
    return table


def Smu2_to_hfs_ratio(table):
    table["Smu2 [D^2]"] /= table["Smu2 [D^2]"].sum()
    table.rename_column("Smu2 [D^2]", "hfs ratio")
    return table


def generate_smart_table(table, hfs=False, hfs_indicator="F"):
    table = select_and_rename_columns(table)
    table = resolve_freq_duplication(table)
    table = resolve_hfs_rot_duplication(table, remove_hfs=not hfs, hfs_indicator=hfs_indicator)
    if hfs:
        table = Smu2_to_hfs_ratio(table)
    return table


def LTEmodel(nu, nu0, g_u, E_u, A_ul, N, Tex, DeltaV, f, Q, Tbg=2.73):
#     Q = Q(Tex)
    N_u = N / Q * g_u * np.exp(-E_u / Tex)

    # lineprofile function
    Deltanu = nu0 / ckms * DeltaV
    phi = (
        np.sqrt(4 * np.log(2) / np.pi)
        / Deltanu
        * np.exp(-4 * np.log(2) * (nu - nu0) ** 2 / Deltanu ** 2)
    )
    tau = (
        ccms ** 2
        / (8 * np.pi * nu ** 2)
        * (np.exp(h * nu / (k * Tex)) - 1.0)
        * A_ul
        * phi
        * N_u
    )

    # intensity
    B_nu = models.BlackBody(temperature=Tex * u.K)
    B_nu_bg = models.BlackBody(temperature=Tbg * u.K)
    
    I = f * (B_nu(nu * u.Hz) - B_nu_bg(nu * u.Hz)) * (1 - np.exp(-tau))
    
    return I, tau




# partition fucntion from classical CDMS database (not the VAMDC ones!)
# CAUTION: this partition function may not be consistenet with the upper state degeneracy
# In particular, you need to check the consistency if you retrieve the spectroscopic data from databases other than CDMS
# Even if the data come from CDMS, you need to check the consistency since partition function calculation could differ between VAMDC and classical CDMS
# TODO: in the future, retrieve all of the data from VAMDC CDMS database using VAMDC-TAP interface; the best is to work vamdclib well, but unfortunately doesn't work well in my environment...
def get_CDMS_partition_function(specie):

    res = requests.get(pf_table_path)

    soup = BeautifulSoup(res.text, "html.parser")

    data = soup.pre.get_text(strip=True).split("\n")
    header = data[0][40:]
    data = data[2:]

    molname_list = [d[7:33] for d in data]
    idx = np.where([specie in molname for molname in molname_list])[0]

    if idx.size == 1:
        idx = idx[0]
    else:
        raise ValueError("Multiple candidates for {:s} found.".format(specie))

    
    # read out the temperature
    T = np.array([float(h.replace("lg(Q(", "").replace("))", "")) for h in header.split()])

    # data
    Q = 10 ** np.array([float(val) if val != "---" else np.nan for val in data[idx][40:].split()])

    print(T, Q)

    # remove nan
    T = T[~np.isnan(Q)]
    Q = Q[~np.isnan(Q)]

    # interpolation
    f = interp1d(T, Q, kind="cubic", fill_value="extrapolate")

    return f





