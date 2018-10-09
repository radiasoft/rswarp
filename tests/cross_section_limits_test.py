import pytest
import math
import sys

import rsoopic.h2crosssections as h2crosssections
from scipy.constants import c, e, m_e, m_p

def test_cross_section_limits():
    """
    Unit test of cross section for electron-impact ionization of hydrogen
    """
    T = 1.e2 # incident energy in eV
    gamma = e * T / (m_e * c**2) + 1.
    v_T = c * math.sqrt(1. - 1. / gamma**2)

    # Calculate cross section using old Rudd BEB model
    sigma = h2crosssections.h2_ioniz_crosssection(v_T)
    if __name__ == '__main__':
        print T, sigma
    assert(9.1e-21 < sigma and 9.2e-21 > sigma)

    # Load module with updated version
    sys.path.insert(1, '/home/vagrant/jupyter/rswarp/rswarp/ionization')
    import crosssections as Xsect
    # Calculate cross section using Kim RBEB model
    h2 = Xsect.H2IonizationEvent()
    sigma = h2.getCrossSection(v_T)
    if __name__ == '__main__':
        print T, sigma
    assert(9.65e-21 < sigma and 9.75e-21 > sigma)

    T = 1.e6 # incident energy in eV
    gamma = e * T / (m_e * c**2) + 1.
    v_T = c * math.sqrt(1. - 1. / gamma**2)

    sigma = h2.getCrossSection(v_T)
    if __name__ == '__main__':
        print T, sigma
    assert(2.3e-23 < sigma and 2.5e-23 > sigma)

    # Calculate cross section using Moller model
    moller = Xsect.IonizationEvent()
    moller.setEps_min(1.2)
    sigma = moller.getCrossSection(v_T)
    if __name__ == '__main__':
        print T, sigma
    assert(2.3e-23 < sigma and 2.5e-23 > sigma)

    gamma = e * T / (m_p * c**2) + 1. # gamma value for incident proton
    #v_T = c * np.sqrt(1. - np.divide(1., gamma**2))
    v_T = c * math.sqrt(1. - 1. / gamma**2)

    # Calculate cross section using Rayzer rule
    p_on_h2 = Xsect.IonIonizationEvent()
    sigma = p_on_h2.getCrossSection(v_T)
    if __name__ == '__main__':
        print T, sigma
    assert(1.3e-21 < sigma and 1.5e-21 > sigma)

    # Calculate cross section using Rudd model
    p_on_h2 = Xsect.RuddIonIonizationEvent()
    sigma = p_on_h2.getCrossSection(v_T)
    if __name__ == '__main__':
        print T, sigma
    assert(3.65e-21 < sigma and 3.85e-21 > sigma)

    #print p_on_h2.eps_min

if __name__ == '__main__':
    test_cross_section_limits()
