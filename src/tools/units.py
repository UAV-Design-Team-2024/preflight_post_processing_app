from enum import Enum, StrEnum

class unit_system(Enum):
    METRIC = 0
    IMPERIAL = 1

class units:

    def __init__(self, unit):
        self.unit_name = unit
        self.force_unit = unit
        self.mass_unit = unit
        self.distance_unit = unit

    @property
    def unit_name(self):
        return self._unit_name
    @unit_name.setter
    def unit_name(self, value):
        if value == 0:
            self._unit_name = "Metric"
        else:
            self._unit_name = "Imperial"

    @property
    def force_unit(self):
        return self._force_unit
    @force_unit.setter
    def force_unit(self, value):
        if value == 0:
            self._force_unit = "N"
        else:
            self._force_unit = "lbf"

    @property
    def mass_unit(self):
        return self._mass_unit

    @mass_unit.setter
    def mass_unit(self, value):
        if value == 0:
            self._mass_unit = "kg"
        else:
            self._mass_unit = "lbs"

    @property
    def distance_unit(self):
        return self._distance_unit
    @distance_unit.setter
    def distance_unit(self, value):
        if value == 0:
            self._distance_unit = "ft"
        else:
            self._distance_unit = "km"
