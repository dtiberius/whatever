from calc import predict_species
from sklearn.datasets import load_iris # used to get data

def test_string():
    species = predict_species(4, 1.8, 0.8, 4.1)
    assert isinstance(species, str)
    
def test_first_observation():
    species = predict_species(5.1, 3.5, 1.4, 0.2)
    assert species == 'setosa'
    
def test_average_setosa():
    species = predict_species(5.006, 3.428, 1.462, 0.246)
    assert species == 'setosa'