from .core import StringIntEnum

# Nomenclature from CHARM: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2944191/
# These areas are defined at different levels of precision - have a look at the CHARM_Key_table
# to find how these 
class Macaque(StringIntEnum):
    primary_motor_cortex = 79
    premotor_cortex = 80
    primary_somatosensory_cortex = 92
