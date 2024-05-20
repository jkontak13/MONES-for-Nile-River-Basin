class ReleaseRangeError(Exception):
    """Raised when we are not able to find a release range from a set of limits"""

    def __init__(self, message, name, current_level, release_limits):
        self.message = message
        self.name = name
        self.current_level = current_level
        self.release_limits = release_limits

    def __str__(self):
        return f"{self.message} \n    - Name: {self.name} \n    - Current water level: {self.current_level} \n    - Release limits: {self.release_limits}"
