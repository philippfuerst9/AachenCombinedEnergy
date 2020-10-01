from icecube import dataio
import numpy as np

#GCD files change 
GCD_2012_path = "/cvmfs/icecube.opensciencegrid.org/data/GCD/GeoCalibDetectorStatus_2012.56063_V0.i3.gz"


def get_string_pos(GCDpath):
    GCD_file = dataio.I3File(GCDpath)
    GCD_file.rewind()
    G_frame = GCD_file.pop_frame()
    omgeo = G_frame.Get("I3Geometry").omgeo

    x = []
    y = []
    z = []
    for i in range(1,79):
        x.append(omgeo[icetray.OMKey(i,1,0)].position.x)
        y.append(omgeo[icetray.OMKey(i,1,0)].position.y)
        z.append(omgeo[icetray.OMKey(i,1,0)].position.y)

    return x,y,z


def DOM_positions(omkeys, omgeo):
    """takes list of I3 OMKeys and I3 omgeo object and returns the positions of all hit DOMs."""
    x_list = []
    y_list = []
    z_list = []
    r_list = []
    for omkey in omkeys:
        x_list.append(omgeo[omkey].position.x)
        y_list.append(omgeo[omkey].position.y)
        z_list.append(omgeo[omkey].position.z)
        r_list.append(np.sqrt(omgeo[omkey].position.x**2 + omgeo[omkey].position.y**2))
    returndict = {"x":x_list,"y":y_list,"z":z_list, "r":r_list}
    return returndict