import numpy as np


class Converter:

    @staticmethod
    def yearly_avg_power_in_twh(power_in_mwh: float) -> float:
        return power_in_mwh / (20 * 1e6)

    @staticmethod
    def deficit_in_bcm_per_year(deficit_in_m: float) -> float:
        return deficit_in_m * 3600 * 24 * 1e-9 / 20

    @staticmethod
    def frequency_of_month_below_min_level(num_of_months_above_level: float) -> float:
        return (240 - num_of_months_above_level) / 240

    @staticmethod
    def covert_datapoint_into_yassin_format(data_point: np.array) -> np.array:
        """
        Assume order of objectives as follows: [ethiopia hydroenergy, sudan deficit, egypt deficit, min HAD level].
        :return: Normalized as in Yassin paper.
        Returned order is [egypt deficit, min HAD level, sudan deficit, ethiopia hydroenergy]
        """
        return np.array([
            np.abs(Converter.deficit_in_bcm_per_year(data_point[2])),
            Converter.frequency_of_month_below_min_level(data_point[3]),
            np.abs(Converter.deficit_in_bcm_per_year(data_point[1])),
            Converter.yearly_avg_power_in_twh(data_point[0]),
        ])

    @staticmethod
    def convert_array(results: np.array) -> np.array:
        result = []
        for dp in results:
            result.append(Converter.covert_datapoint_into_yassin_format(dp))
        return np.array(result)
