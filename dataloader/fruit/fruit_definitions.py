import os
from abc import ABC, abstractmethod
from enum import Enum

from camera_definitions import CameraType
from ios.envi import load_envi, load_referenced_envi


class Fruit(Enum):
    AVOCADO = 'Avocado'
    KIWI = 'Kiwi'
    MANGO = 'Mango'
    KAKI = 'Kaki'
    PAPAYA = 'Papaya'
    ALL = 'All'

def str2fruit(s: str):
    if s.lower() == 'avocado':
        return Fruit.AVOCADO
    elif s.lower() == 'kiwi':
        return Fruit.KIWI
    elif s.lower() == 'mango':
        return Fruit.MANGO
    elif s.lower() == 'kaki':
        return Fruit.KAKI
    elif s.lower() == 'papaya':
        return Fruit.PAPAYA
    elif s.lower() == 'all':
        return Fruit.ALL
    else:
        raise Exception('{} is not a valid fruit'.format(s))

class ClassificationType(Enum):
    RIPENESS = 'ripeness'
    FIRMNESS = 'firmness'
    SUGAR = 'sugar'

def str2classification_type(s: str):
    if s.lower() == 'ripeness':
        return ClassificationType.RIPENESS
    elif s.lower() == 'firmness':
        return ClassificationType.FIRMNESS
    elif s.lower() == 'sugar':
        return ClassificationType.SUGAR
    else:
        raise Exception('{} is not a valid classification_type'.format(s))

class Side(Enum):
    FRONT = 'front'
    BACK = 'back'

class ID(Enum):
    UNKNOWN = '?'
    ID_1 = '01'
    ID_2 = '02'
    ID_3 = '03'
    ID_4 = '04'
    ID_5 = '05'
    ID_6 = '06'
    ID_7 = '07'
    ID_8 = '08'
    ID_9 = '09'
    ID_10 = '10'
    ID_11 = '11'
    ID_12 = '12'
    ID_13 = '13'
    ID_14 = '14'
    ID_15 = '15'
    ID_16 = '16'
    ID_17 = '17'
    ID_18 = '18'
    ID_19 = '19'
    ID_20 = '20'
    ID_21 = '21'
    ID_22 = '22'
    ID_23 = '23'
    ID_24 = '24'
    ID_25 = '25'
    ID_26 = '26'
    ID_27 = '27'
    ID_28 = '28'
    ID_29 = '29'
    ID_30 = '30'
    ID_31 = '31'
    ID_32 = '32'
    ID_33 = '33'
    ID_34 = '34'
    ID_35 = '35'
    ID_36 = '36'
    ID_37 = '37'
    ID_38 = '38'
    ID_39 = '39'
    ID_40 = '40'
    ID_41 = '41'
    ID_42 = '42'
    ID_43 = '43'
    ID_44 = '44'
    ID_45 = '45'
    ID_46 = '46'
    ID_47 = '47'
    ID_48 = '48'
    ID_49 = '49'
    ID_50 = '50'
    ID_51 = '51'
    ID_52 = '52'
    ID_53 = '53'
    ID_54 = '54'
    ID_55 = '55'
    ID_56 = '56'
    ID_57 = '57'
    ID_58 = '58'
    ID_59 = '59'
    ID_60 = '60'
    ID_61 = '61'
    ID_62 = '62'
    ID_63 = '63'
    ID_64 = '64'
    ID_65 = '65'
    ID_66 = '66'
    ID_67 = '67'
    ID_68 = '68'
    ID_69 = '69'
    ID_70 = '70'
    ID_71 = '71'
    ID_72 = '72'
    ID_73 = '73'
    ID_74 = '74'
    ID_75 = '75'
    ID_76 = '76'
    ID_77 = '77'
    ID_78 = '78'
    ID_79 = '79'
    ID_80 = '80'
    ID_81 = '81'
    ID_82 = '82'
    ID_83 = '83'
    ID_84 = '84'
    ID_85 = '85'
    ID_86 = '86'
    ID_87 = '87'
    ID_88 = '88'
    ID_89 = '89'
    ID_90 = '90'
    ID_91 = '91'
    ID_92 = '92'
    ID_93 = '93'
    ID_94 = '94'
    ID_95 = '95'
    ID_96 = '96'
    ID_97 = '97'
    ID_98 = '98'
    ID_99 = '99'

class Day(Enum):
    TEST_1 = 'test'
    TEST_2 = 'test2'
    DAY_1 = 'day_01'
    DAY_2 = 'day_02'
    DAY_3 = 'day_03'
    DAY_4 = 'day_04'
    DAY_5 = 'day_05'
    DAY_6 = 'day_06'
    DAY_7 = 'day_07'
    DAY_8 = 'day_08'
    DAY_9 = 'day_09'
    DAY_10 = 'day_10'
    DAY_11 = 'day_11'
    DAY_M2_1 = 'day_m2_01'
    DAY_M2_2 = 'day_m2_02'
    DAY_M2_3 = 'day_m2_03'
    DAY_M2_4 = 'day_m2_04'
    DAY_M2_5 = 'day_m2_05'
    DAY_M2_6 = 'day_m2_06'
    DAY_M2_7 = 'day_m2_07'
    DAY_M2_8 = 'day_m2_08'
    DAY_M2_9 = 'day_m2_09'
    DAY_M2_10 = 'day_m2_10'
    DAY_M2_11 = 'day_m2_11'
    DAY_M2_12 = 'day_m2_12'
    DAY_M2_13 = 'day_m2_13'
    DAY_M2_14 = 'day_m2_14'
    DAY_M2_15 = 'day_m2_15'
    DAY_M2_16 = 'day_m2_16'
    DAY_M2_17 = 'day_m2_17'
    DAY_M3_1 = 'day_1_m3'
    DAY_M3_2 = 'day_2_m3'
    DAY_M3_3 = 'day_3_m3'
    DAY_M3_4 = 'day_4_m3'
    DAY_M3_5 = 'day_5_m3'
    DAY_M3_6 = 'day_6_m3'
    DAY_M3_7 = 'day_7_m3'
    DAY_M3_8 = 'day_8_m3'
    DAY_M3_9 = 'day_9_m3'
    DAY_M3_10 = 'day_10_m3'
    DAY_M3_11 = 'day_11_m3'
    DAY_M3_12 = 'day_12_m3'
    DAY_M4_1 = 'day_m4_01'
    DAY_M4_2 = 'day_m4_02'
    DAY_M4_3 = 'day_m4_03'
    DAY_M4_4 = 'day_m4_04'
    DAY_M4_5 = 'day_m4_05'
    DAY_M4_6 = 'day_m4_06'
    DAY_M4_7 = 'day_m4_07'
    DAY_M4_8 = 'day_m4_08'
    DAY_M4_9 = 'day_m4_09'

class FirmnessLevel(Enum):
    TOO_HARD = 'too_hard'
    READY = 'ready'
    TOO_SOFT = 'too_soft'
    UNKNOWN = None

class SugarLevel(Enum):
    NOT_SWEET = 'not_sweet'
    READY = 'ready'
    TOO_SWEET = 'too_sweet'
    UNKNOWN = None

class RipenessState(Enum):
    UNRIPE = 'unripe'
    PERFECT = 'perfect'
    RIPE = 'ripe'
    NEAR_OVERRIPE = 'near_overripe'
    OVERRIPE = 'overripe'
    UNKNOWN = None


class FruitRecord:
    def __init__(self, fruit: Fruit, side: Side, day: Day, id: ID, camera_type: CameraType, label: str = None, origin=None):
        self.fruit = fruit
        self.id = id
        self.side = side
        self.day = day
        self.camera_type = camera_type
        self.label = label
        self.origin = origin

    def get_file_path(self):
        name = self.fruit.value.lower() + "_" + self.day.value.lower() + "_" + self.id.value + "_" + self.side.value.lower()
        file_name = os.path.join(self.fruit.value.capitalize(), self.camera_type.value.upper(), self.day.value.lower(), name)
        return file_name

    def get_name(self):
        name = self.fruit.value.lower() + "_" + self.day.value.lower() + "_" + self.id.value + \
           "_" + self.side.value.lower() + "_" + self.camera_type.value.lower()
        return name

    def load(self, _origin=None, is_already_referenced=False):
        file_path = self.get_file_path()

        if is_already_referenced:
            header, data = load_envi(file_path, _origin)
        else:
            header, data = load_referenced_envi(file_path, _origin)

        return header, data

    def is_labeled(self):
        return self.label is not None

    def __str__(self):
        if self.is_labeled():
            return f"{self.fruit.value}: {self.id.value}/{self.side.value} \n" \
                f"\ton day: {self.day.value}\n" \
                f"\trecored with: {self.camera_type.value}\n" \
                f"\thas state: {str(self.label)}" \
                f"\tpath: {self.get_file_path()}"
        else:
            return f"{self.fruit.value}: {self.id.value}/{self.side.value} \n" \
                f"\ton day: {self.day.value}\n" \
                f"\trecored with: {self.camera_type.value}\n" \
                f"\tpath: {self.get_file_path()}"

    def __eq__(self, other: Fruit):
        if not isinstance(other, FruitRecord):
            return False
        return (self.fruit == other.fruit) and (self.camera_type == other.camera_type) and (self.side == other.side) and \
               (self.id == other.id) and (self.day == other.day)

    def __hash__(self):
        return hash(tuple(sorted(self.__dict__.items())))


class AvocadoLabel:
    def __init__(self, _init_weight: int, _end_weight: int, _storage_days: int, _firmness: int,
                 _ripeness_state: RipenessState, _comment: str = None):
        self.init_weight = _init_weight
        self.end_weight = _end_weight
        self.storage_days = _storage_days
        self.firmness = _firmness  # in g / cm^2
        self.ripeness_state = _ripeness_state
        self.comment = _comment

    def __str__(self):
        if self.comment is not None:
            return "{%s, firmness: %s, comment: '%s'}" % (
                self.ripeness_state.value, self.firmness, self.comment)
        else:
            return "{%s, firmness: %s}" % (
                self.ripeness_state.value, self.firmness)

    def get_firmness_level(self) -> FirmnessLevel:
        if self.firmness is None:
            return FirmnessLevel.UNKNOWN
        elif self.firmness > 1200:
            return FirmnessLevel.TOO_HARD
        elif self.firmness < 900:
            return FirmnessLevel.TOO_SOFT
        else:
            return FirmnessLevel.READY


class SweetFruitLabel(ABC):
    def __init__(self, _init_weight: int, _end_weight: int, _storage_days: int, _firmness: int, _sugar_content: float,
                 _ripeness_state: RipenessState, _comment: str = None):
        self.init_weight = _init_weight
        self.end_weight = _end_weight
        self.storage_days = _storage_days
        self.firmness = _firmness  # in g / cm^2
        self.sugar_content = _sugar_content  # in °Brix
        self.ripeness_state = _ripeness_state
        self.comment = _comment

    def __str__(self):
        if self.comment is not None:
            return "{%s, firmness: %s, sugar content: %s, comment: '%s'}" % (
                self.ripeness_state.value, self.firmness, self.sugar_content, self.comment)
        else:
            return "{%s, firmness: %s, sugar content: %s}" % (
                self.ripeness_state.value, self.firmness, self.sugar_content)

    @abstractmethod
    def get_firmness_level(self) -> FirmnessLevel:
        pass

    @abstractmethod
    def get_sugar_level(self) -> SugarLevel:
        pass


class KiwiLabel(SweetFruitLabel):
    def __init__(self, _init_weight: int, _end_weight: int, _storage_days: int, _firmness: int, _sugar_content: float,
                 _ripeness_state: RipenessState, _comment: str = None):
        super().__init__(_init_weight, _end_weight, _storage_days, _firmness, _sugar_content, _ripeness_state,
                         _comment)

    def get_firmness_level(self) -> FirmnessLevel:
        if self.firmness is None:
            return FirmnessLevel.UNKNOWN
        elif self.firmness > 1500:
            return FirmnessLevel.TOO_HARD
        elif self.firmness < 1000:
            return FirmnessLevel.TOO_SOFT
        else:
            return FirmnessLevel.READY

    def get_sugar_level(self) -> SugarLevel:
        if self.sugar_content is None:
            return SugarLevel.UNKNOWN
        elif self.sugar_content < 15.5:
            return SugarLevel.NOT_SWEET
        elif self.sugar_content > 17:
            return SugarLevel.TOO_SWEET
        else:
            return SugarLevel.READY


class KakiLabel(SweetFruitLabel):
    def __init__(self, _init_weight: int, _end_weight: int, _storage_days: int, _firmness: int, _sugar_content: float,
                 _ripeness_state: RipenessState, _comment: str = None):
        super().__init__(_init_weight, _end_weight, _storage_days, _firmness, _sugar_content, _ripeness_state,
                         _comment)

    def get_sugar_level(self) -> SugarLevel:
        if self.sugar_content is None:
            return SugarLevel.UNKNOWN
        elif self.sugar_content < 20:
            return SugarLevel.NOT_SWEET
        elif self.sugar_content > 22:
            return SugarLevel.TOO_SWEET
        else:
            return SugarLevel.READY

    def get_firmness_level(self) -> FirmnessLevel:
        if self.firmness is None:
            return FirmnessLevel.UNKNOWN
        elif self.firmness > 1500:
            return FirmnessLevel.TOO_HARD
        elif self.firmness < 500:
            return FirmnessLevel.TOO_SOFT
        else:
            return FirmnessLevel.READY


class MangoLabel(SweetFruitLabel):
    def __init__(self, _init_weight: int, _end_weight: int, _storage_days: int, _firmness: int, _sugar_content: float,
                 _ripeness_state: RipenessState, _comment: str = None):
        super().__init__(_init_weight, _end_weight, _storage_days, _firmness, _sugar_content, _ripeness_state,
                         _comment)

    def get_sugar_level(self) -> SugarLevel:
        if self.sugar_content is None:
            return SugarLevel.UNKNOWN
        elif self.sugar_content < 14:
            return SugarLevel.NOT_SWEET
        elif self.sugar_content > 16:
            return SugarLevel.TOO_SWEET
        else:
            return SugarLevel.READY

    def get_firmness_level(self) -> FirmnessLevel:
        if self.firmness is None:
            return FirmnessLevel.UNKNOWN
        elif self.firmness > 12500:
            return FirmnessLevel.TOO_HARD
        elif self.firmness < 5000:
            return FirmnessLevel.TOO_SOFT
        else:
            return FirmnessLevel.READY


class PapayaLabel(SweetFruitLabel):
    def __init__(self, _init_weight: int, _end_weight: int, _storage_days: int, _firmness: int, _sugar_content: float,
                 _ripeness_state: RipenessState, _comment: str = None):
        super().__init__(_init_weight, _end_weight, _storage_days, _firmness, _sugar_content, _ripeness_state,
                         _comment)

    def get_sugar_level(self) -> SugarLevel:
        if self.sugar_content is None:
            return SugarLevel.UNKNOWN
        elif self.sugar_content < 11:
            return SugarLevel.NOT_SWEET
        elif self.sugar_content > 13:
            return SugarLevel.TOO_SWEET
        else:
            return SugarLevel.READY

    def get_firmness_level(self) -> FirmnessLevel:
        if self.firmness is None:
            return FirmnessLevel.UNKNOWN
        elif self.firmness > 1800:
            return FirmnessLevel.TOO_HARD
        elif self.firmness < 800:
            return FirmnessLevel.TOO_SOFT
        else:
            return FirmnessLevel.READY