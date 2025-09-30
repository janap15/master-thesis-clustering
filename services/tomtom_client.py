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

    def _generate_matrix_routing_request_body(self, origins, destinations):
        origins = [{"point": {"latitude": origin.latitude, "longitude": origin.longitude}} for origin in origins]
        destinations = [{"point": {"latitude": destination.latitude, "longitude": destination.longitude}} for destination in destinations]
        return {"origins": origins, "destinations": destinations}

    def _submit_matrix_routing_request(self, origins, destinations):
        request_body = json.dumps(self._generate_matrix_routing_request_body(origins, destinations))
        url = f"{self.matrix_routing_base_url}?key={self.api_key}&routeType=fastest&travelMode=truck"
        response = requests.post(url, headers=self.headers, data=request_body)

        if response.status_code != 202:
            raise RuntimeError(f"An error occurred during Matrix routing submission request: {response.status_code}")

        print(f"Matrix routing request submitted: {response.json()}")
        return response.json()["jobId"]

    def _poll_matrix_routing_result(self, job_id: str):
        status_url = f"{self.matrix_routing_base_url}/{job_id}?key={self.api_key}"
        download_url = f"{self.matrix_routing_base_url}/{job_id}/result?key={self.api_key}"

        while True:
            status_response = requests.get(status_url)
            if status_response.status_code != 200:
                raise RuntimeError(f"An error occurred during Matrix routing request: {status_response.status_code}")
            state = status_response.json()["state"]
            if state == "Completed":
                break
            elif state == "Failed":
                raise RuntimeError(f"An error occurred during Matrix routing request: {status_response.json()}")
            time.sleep(2)
            print("sleeping...")

        return requests.get(download_url)

    def _response_to_result_matrix(self, response: requests.Response, m: int, n: int):
        data = response.json()["data"]

        distance_matrix = [[0] * n for _ in range(m)]

        for row in data:
            origin_idx = row["originIndex"]
            dest_idx = row["destinationIndex"]
            distance_matrix[origin_idx][dest_idx] = row["routeSummary"]["lengthInMeters"]

        print(f"Distance matrix: {distance_matrix}")
        return distance_matrix

    def get_distance_matrix(self, origins, destinations):
        job_id = self._submit_matrix_routing_request(origins, destinations)
        response = self._poll_matrix_routing_result(job_id)
        return self._response_to_result_matrix(response, len(origins), len(destinations))
