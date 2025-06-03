# circlefitanalysis.py
# find where the MS effect is minimised (ie where the recurl hits occur near phi= pi ±0.5 rad)
import numpy as np

def fit_circle(x, y):
    x = np.array(x)
    y = np.array(y)
    x_m = np.mean(x)
    y_m = np.mean(y)
    
    u = x - x_m
    v = y - y_m
    Suu = np.sum(u*u)
    Suv = np.sum(u*v)
    Svv = np.sum(v*v)
    Suuu = np.sum(u*u*u)
    Svvv = np.sum(v*v*v)
    Suvv = np.sum(u*v*v)
    Svuu = np.sum(v*u*u)
    
    A = np.array([[Suu, Suv], [Suv, Svv]])
    B = 0.5 * np.array([Suuu + Suvv, Svvv + Svuu])
    uc, vc = np.linalg.solve(A, B)
    
    xc = x_m + uc
    yc = y_m + vc
    R = np.mean(np.sqrt((x - xc)**2 + (y - yc)**2))
    return xc, yc, R

def recurl_angle(track_hits, tol=0.1):
    """
    Given track_hits: list of (x,y,z) triples, find the
    angle between first and last hit about the fitted circle centre.
    Returns wrapped delta_phi in (-pi,pi].
    """
    # extract transverse coords
    pts = np.array(track_hits)
    x, y = pts[:,0], pts[:,1]
    # fit circle
    x0,y0,R = fit_circle(x, y)
    # compute phi of first and last hits
    phi0 = np.arctan2(y[0]-y0, x[0]-x0)
    phi1 = np.arctan2(y[-1]-y0, x[-1]-x0)
    # raw difference
    dphi = phi1 - phi0
    # wrap into (-pi,pi]
    dphi = (dphi + np.pi) % (2*np.pi) - np.pi
    return dphi
