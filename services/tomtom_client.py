import json
import math
import time

import requests


class TomTomClient:
    def __init__(self, api_key):
        self.api_key = api_key
        self.matrix_routing_base_url = "https://api.tomtom.com/routing/matrix/2/async"
        self.headers = {
            "Content-Type": "application/json",
        }

    def generate_matrix_routing_request_body(self, packages):
        origins = [{"point": {"latitude": p.latitude, "longitude": p.longitude}} for p in packages]
        destinations = [{"point": {"latitude": p.latitude, "longitude": p.longitude}} for p in packages]
        return {"origins": origins, "destinations": destinations}

    def submit_matrix_routing_request(self, packages):
        request_body = json.dumps(self.generate_matrix_routing_request_body(packages))
        url = f"{self.matrix_routing_base_url}?key={self.api_key}&routeType=fastest&travelMode=truck"
        response = requests.post(url, headers=self.headers, data=request_body)

        if response.status_code != 202:
            raise RuntimeError(f"An error occurred during Matrix routing submission request: {response.status_code}")

        print(f"Matrix routing request submitted: {response.json()}")
        return response.json()["jobId"]

    def poll_matrix_routing_result(self, job_id: str):
        status_url = f"{self.matrix_routing_base_url}/{job_id}?key={self.api_key}"
        download_url = f"{self.matrix_routing_base_url}/{job_id}/result?key={self.api_key}"

        while True:
            status_response = requests.get(status_url)
            if status_response.status_code != 200:
                raise RuntimeError(f"An error occurred during Matrix routing request: {status_response.status_code}")
            if status_response.json().get("state") == "Completed":
                break
            time.sleep(2)
            print("sleeping...")

        return requests.get(download_url)

    def response_to_result_matrix(self, response: requests.Response):
        data = response.json()["data"]

        n = int(math.sqrt(len(data)))
        distance_matrix = [[0] * n for _ in range(n)]

        for row in data:
            distance_matrix[row["originIndex"]][row["destinationIndex"]] = row["routeSummary"]["lengthInMeters"]

        print(f"Distance matrix: {distance_matrix}")
        return distance_matrix
