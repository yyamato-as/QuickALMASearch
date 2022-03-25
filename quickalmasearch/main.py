import alminer
from astropy.coordinates import SkyCoord
from astroquery.splatalogue import Splatalogue
import astropy.units as u
from helpers import generate_smart_table

arcsec = 1. / 60. # in arcmin


def get_line_query(species, freq_range=(84*u.GHz, 950*u.GHz), line_list=["CDMS"], hfs=False):
    line_queries = {}
    for s in species:
        chemical_name = " {:s} ".format(s)
        q = Splatalogue.query_lines(
            *freq_range,
            chemical_name=chemical_name,
            line_lists=line_list,#, "JPL"],
            show_upper_degeneracy=True,
            # line_strengths=["ls4"],
            # energy_type="eu_k",
        )
        line_queries[s] = generate_smart_table(q, hfs=hfs)

        print("{} lines found for {}.".format(len(q), s))

    return line_queries


def get_line_observations(radec, line_queries, search_radius=5.0, conesearch_kwargs={"public": None}):
    c = SkyCoord(radec, frame="icrs")
    print("Searching within {} arcsec region around {}...".format(search_radius, radec))
    search_radius *= arcsec
    q = alminer.conesearch(ra=c.ra.deg, dec=c.dec.deg, search_radius=search_radius, **conesearch_kwargs)

    line_obs = {}

    for s in line_queries.keys():
        for line in line_queries[s]:
            line_name = str(line["Species"]) + " " + str(line["QNs"])
            print("Searching for {:s} at {:s} GHz...".format(line_name, line["nu0 [GHz]"]))
            lines = alminer.line_coverage(q, line_freq=line["nu0 [GHz]"], line_name=line_name, print_summary=False)
            if not lines.empty:
                print("Archival data found for {:s}.".format(line_name))
                print(lines)
                line_obs[line_name] = {"observations": lines, "freq": line["nu0 [GHz]"]}
            else:
                print("No archival data found.")

    print("{} observations found in total.".format(len(line_obs)))

    return line_obs