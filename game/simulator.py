import requests
import json


class BaseSimulator(object):
    def __init__(self, host: str = 'http://localhost:8080', **kwargs):
        self.host = host

    def load(self, level_index: int, seed: int) -> any:
        return self._do_request('/load', {
            'seed': seed,
            'levelIndex': level_index
        })

    def click(self, state: object, x, y) -> object:
        return self._do_request('/click', {
            'state': state,
            'x': x,
            'y': y
        })

    def session_create(self, state: any, return_full_state: bool = False) -> any:
        return self._do_request('/session/create', {
            'state': state,
            'returnFullState': return_full_state
        })

    def session_click(self, session_id: str, x: int, y: int, return_full_state: bool = False) -> any:
        try:
            result = self._do_request('/session/click', {
                'sessionId': session_id,
                'x': x,
                'y': y,
                'returnFullState': return_full_state
            })
        except requests.HTTPError as e:  # Workaround because error sometimes occurs
            print(e)
            result = self._do_request('/session/click', {
                'sessionId': session_id,
                'x': x,
                'y': y,
                'returnFullState': return_full_state
            })

        return result

    def session_destroy(self, session_id: str) -> any:
        return self._do_request('/session/destroy', {
            'sessionId': session_id
        })

    def session_status(self, session_id: str) -> any:
        return self._do_request('/session/status', {
            'sessionId': session_id
        })

    def sessions_list(self) -> any:
        return self._do_request('/sessions/list', {})

    def sessions_clear(self) -> any:
        return self._do_request('/sessions/clear', {})

    def _do_request(self, url: str, json_payload: any) -> any:
        headers = {"content-type": "application/json"}
        response = requests.post(self.host + url, data=json.dumps(json_payload), headers=headers, timeout=100)
        response.raise_for_status()
        return response.json()


class Simulator(BaseSimulator):

    def __init__(self, host: str, **kwargs):
        super().__init__(host, **kwargs)
        self.session_id = None
        self.return_full_states = kwargs.get('return_full_state', False)

    def reset(self, level: int, seed: int) -> dict:
        if self.session_id is not None:
            self.session_destroy(self.session_id)

        try:
            created_level = self.load(level, seed)
            session_result = self.session_create(json.loads(created_level['state']), self.return_full_states)
        except requests.exceptions.HTTPError:
            created_level = {}
            session_result = {}

        self.session_id = session_result.get('sessionId', None)

        return created_level

    def step(self, x: int, y: int, **kwargs) -> dict:
        try:
            results = self.session_click(
                kwargs.get('session_id', self.session_id),
                x,
                y,
                self.return_full_states)
        except requests.exceptions.HTTPError:
            results = {}
        return results

    def close(self):
        self.session_destroy(self.session_id)