class Package:

    def __init__(self, package_id, latitude, longitude, priority=0, opening_hour="", closing_hour="", cluster=-1,
                 closest_package=None):
        self.package_id = package_id
        self.latitude = latitude
        self.longitude = longitude
        self.priority = priority
        self.opening_hour = opening_hour
        self.closing_hour = closing_hour
        self.cluster = cluster
        self.closest_package = closest_package

    def get_id(self):
        return self.package_id

    def set_id(self, package_id):
        self.package_id = package_id

    def get_latitude(self):
        return self.latitude

    def set_latitude(self, latitude):
        self.latitude = latitude

    def get_longitude(self):
        return self.longitude

    def set_longitude(self, longitude):
        self.longitude = longitude

    def get_priority(self):
        return self.priority

    def set_priority(self, priority):
        self.priority = priority

    def get_opening_hour(self):
        return self.opening_hour

    def get_closing_hour(self):
        return self.closing_hour

    def get_cluster(self):
        return self.cluster

    def set_cluster(self, cluster):
        self.cluster = cluster

    def __str__(self):
        data = (f"Package: id={self.package_id}, lat={self.latitude}, lon={self.longitude}, priority={self.priority}, "
                f"opening_hour={self.opening_hour}, closing_hour={self.closing_hour}, cluster={self.cluster}")
        if self.closest_package:
            data += f", closest_package={self.closest_package}"
        return data

    def to_dict(self):
        data = {"point": {"latitude": self.latitude, "longitude": self.longitude}}
        if self.priority:
            data["priority"] = self.priority
        if self.opening_hour and self.closing_hour:
            data["timeWindow"] = [{"opening_hour": self.opening_hour, "closing_hour": self.closing_hour}]
        return data