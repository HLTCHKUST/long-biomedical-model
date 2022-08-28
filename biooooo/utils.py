import datasets
from bigbio.dataloader import BigBioConfigHelpers

# Manually curated from https://docs.google.com/spreadsheets/d/1Uq6dJXi9qP43_yM3zgbftPpgsI7t7czmQ-rc8cnuW04/edit?usp=sharing
dataset_config_names = [
    'cellfinder_bigbio_kb',
    'cellfinder_splits_bigbio_kb',
    'chebi_nactem_fullpaper_bigbio_kb',
    'cord_ner_bigbio_kb',
    'linnaeus_bigbio_kb',
    'linnaeus_filtered_bigbio_kb',
    'nlmchem_bigbio_kb',
    'nlmchem_bigbio_text',
    'pmc_patients_bigbio_pairs',
    'spl_adr_200db_train_bigbio_kb'
]

# Load all curated datasets
def load_curated_dataset():
    conhelps = BigBioConfigHelpers()
    bb_public_helpers = conhelps.filtered(
        lambda x: x.config.name in dataset_config_names
    )

    bb_public_datasets = {}
    for helper in bb_public_helpers:
        try:
            print(f'Loading `{helper.config.name}`')
            bb_public_datasets[helper.config.name] = helper.load_dataset()
        except:
            print(f'Failed loading `{helper.config.name}`')
            
    return bb_public_datasets