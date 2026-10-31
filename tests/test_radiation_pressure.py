import numpy as np
import pytest
from qrc_pulsar_bench.radiation_pressure import OpticalPlate,PlateRadiationPressure
from qrc_pulsar_bench.cislunar_forces import AU,SOLAR_PRESSURE_1AU


@pytest.mark.parametrize('specular,diffuse,factor',[(0,0,1),(1,0,2),(0,1,5/3)])
def test_plate_momentum_absorbed_specular_and_diffuse(specular,diffuse,factor):
    r=np.array([2e6,0,0]); earth=np.array([0,4e8,0]); sun=r+[AU,0,0]
    model=PlateRadiationPressure(100,[OpticalPlate(2,(1,0,0),specular,diffuse)],lambda t:np.eye(3),source='synthetic-control',available_at_s=0)
    np.testing.assert_allclose(model(r,earth,sun,0),[-2*SOLAR_PRESSURE_1AU*factor/100,0,0],rtol=1e-14)


def test_edge_on_and_back_facing_plates_are_unilluminated():
    r=np.array([2e6,0,0]); earth=np.array([0,4e8,0]); sun=r+[AU,0,0]
    for normal in [(0,1,0),(-1,0,0)]:
        model=PlateRadiationPressure(100,[OpticalPlate(2,normal)],lambda t:np.eye(3),source='test',available_at_s=0)
        np.testing.assert_array_equal(model(r,earth,sun,0),np.zeros(3))
    with pytest.raises(ValueError): OpticalPlate(1,(1,0,0),.7,.7)
