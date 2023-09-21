VALID_DEBRIS_DATASET_CONFIG = {
    'debris/SPECIM_FX10/objectwise',
    'debris/CORNING_HSI/objectwise',
    'debris/SPECIM_FX10/patchwise',
    'debris/CORNING_HSI/patchwise',
}

VALID_FRUIT_DATASET_CONFIG = {
    'fruit/avocado/ripeness/INNOSPEC_REDEYE',
    'fruit/avocado/firmness/INNOSPEC_REDEYE',
    'fruit/kiwi/ripeness/INNOSPEC_REDEYE',
    'fruit/kiwi/firmness/INNOSPEC_REDEYE',
    'fruit/kiwi/sugar/INNOSPEC_REDEYE',
    'fruit/avocado/ripeness/SPECIM_FX10',
    'fruit/avocado/firmness/SPECIM_FX10',
    'fruit/kiwi/ripeness/SPECIM_FX10',
    'fruit/kiwi/firmness/SPECIM_FX10',
    'fruit/kiwi/sugar/SPECIM_FX10',
    'fruit/mango/ripeness/SPECIM_FX10',
    'fruit/mango/firmness/SPECIM_FX10',
    'fruit/mango/sugar/SPECIM_FX10',
    'fruit/kaki/ripeness/SPECIM_FX10',
    'fruit/kaki/firmness/SPECIM_FX10',
    'fruit/kaki/sugar/SPECIM_FX10',
    'fruit/papaya/ripeness/SPECIM_FX10',
    'fruit/papaya/firmness/SPECIM_FX10',
    'fruit/papaya/sugar/SPECIM_FX10',
    'fruit/avocado/ripeness/CORNING_HSI',
    'fruit/avocado/firmness/CORNING_HSI',
    'fruit/mango/ripeness/CORNING_HSI',
    'fruit/mango/firmness/CORNING_HSI',
    'fruit/mango/sugar/CORNING_HSI',
    'fruit/kaki/ripeness/CORNING_HSI',
    'fruit/kaki/firmness/CORNING_HSI',
    'fruit/kaki/sugar/CORNING_HSI',
    'fruit/papaya/ripeness/CORNING_HSI',
    'fruit/papaya/firmness/CORNING_HSI',
    'fruit/papaya/sugar/CORNING_HSI',
}

VALID_HRSS_DATASET_CONFIG = {
    'remote_sensing/indian_pines/0.3',
    'remote_sensing/indian_pines/0.1',
    'remote_sensing/indian_pines/0.05',
    'remote_sensing/salinas/0.3',
    'remote_sensing/salinas/0.1',
    'remote_sensing/salinas/0.05',
    'remote_sensing/paviaU/0.3',
    'remote_sensing/paviaU/0.1',
    'remote_sensing/paviaU/0.05',
}


VALID_DATASET_CONFIG = VALID_FRUIT_DATASET_CONFIG | VALID_DEBRIS_DATASET_CONFIG | VALID_HRSS_DATASET_CONFIG

