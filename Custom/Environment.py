import tensorforce
from tensorforce import Environment, TensorforceError
import importlib
import json
import os
import sys
from threading import Thread
import time
from traceback import format_tb

from tensorforce import TensorforceError, util
import tensorforce.environments
from tensorforce.environments import RemoteEnvironment, OpenAIGym

from game.lilys_garden_env import LilysGardenEnv


class Environment2(Environment):

    @staticmethod
    def create(
            environment=None, max_episode_timesteps=None, remote=None, blocking=False, host=None,
            port=None, **kwargs
    ):
        """
                Creates an environment from a specification. In case of "socket-server" remote mode, runs
                environment in server communication loop until closed.

                Args:
                    environment (specification | Environment class/object): JSON file, specification key,
                        configuration dictionary, library module, `Environment` class/object, or gym.Env
                        (<span style="color:#C00000"><b>required</b></span>, invalid for "socket-client"
                        remote mode).
                    max_episode_timesteps (int > 0): Maximum number of timesteps per episode, overwrites
                        the environment default if defined
                        (<span style="color:#00C000"><b>default</b></span>: environment default, invalid
                        for "socket-client" remote mode).
                    remote ("multiprocessing" | "socket-client" | "socket-server"): Communication mode for
                        remote environment execution of parallelized environment execution, "socket-client"
                        mode requires a corresponding "socket-server" running, and "socket-server" mode
                        runs environment in server communication loop until closed
                        (<span style="color:#00C000"><b>default</b></span>: local execution).
                    blocking (bool): Whether remote environment calls should be blocking
                        (<span style="color:#00C000"><b>default</b></span>: not blocking, invalid unless
                        "multiprocessing" or "socket-client" remote mode).
                    host (str): Socket server hostname or IP address
                        (<span style="color:#C00000"><b>required</b></span> only for "socket-client" remote
                        mode).
                    port (int): Socket server port
                        (<span style="color:#C00000"><b>required</b></span> only for "socket-client/server"
                        remote mode).
                    kwargs: Additional arguments.
                """
        if remote not in ('multiprocessing', 'socket-client'):
            if blocking:
                raise TensorforceError.invalid(
                    name='Environment.create', argument='blocking',
                    condition='no multiprocessing/socket-client instance'
                )
        if remote not in ('socket-client', 'socket-server'):
            if host is not None:
                raise TensorforceError.invalid(
                    name='Environment.create', argument='host', condition='no socket instance'
                )
            elif port is not None:
                raise TensorforceError.invalid(
                    name='Environment.create', argument='port', condition='no socket instance'
                )

        if remote == 'multiprocessing':
            from tensorforce.environments import MultiprocessingEnvironment
            environment = MultiprocessingEnvironment(
                blocking=blocking, environment=environment,
                max_episode_timesteps=max_episode_timesteps, **kwargs
            )
            return environment

        elif remote == 'socket-client':
            if environment is not None:
                raise TensorforceError.invalid(
                    name='Environment.create', argument='environment',
                    condition='socket-client instance'
                )
            elif max_episode_timesteps is not None:
                raise TensorforceError.invalid(
                    name='Environment.create', argument='max_episode_timesteps',
                    condition='socket-client instance'
                )
            elif len(kwargs) > 0:
                raise TensorforceError.invalid(
                    name='Environment.create', argument='kwargs',
                    condition='socket-client instance'
                )
            from tensorforce.environments import SocketEnvironment
            environment = SocketEnvironment(host=host, port=port, blocking=blocking)
            return environment

        elif remote == 'socket-server':
            from tensorforce.environments import SocketEnvironment
            SocketEnvironment.remote(
                port=port, environment=environment, max_episode_timesteps=max_episode_timesteps,
                **kwargs
            )

        elif isinstance(environment, (EnvironmentWrapper2, RemoteEnvironment)):
            if max_episode_timesteps is not None:
                raise TensorforceError.invalid(
                    name='Environment.create', argument='max_episode_timesteps',
                    condition='EnvironmentWrapper instance'
                )
            if len(kwargs) > 0:
                raise TensorforceError.invalid(
                    name='Environment.create', argument='kwargs',
                    condition='EnvironmentWrapper instance'
                )
            return environment

        elif isinstance(environment, type) and \
                issubclass(environment, (EnvironmentWrapper2, RemoteEnvironment)):
            raise TensorforceError.type(
                name='Environment.create', argument='environment', dtype=type(environment)
            )

        elif isinstance(environment, Environment):
            return EnvironmentWrapper2(
                environment=environment, max_episode_timesteps=max_episode_timesteps
            )

        elif isinstance(environment, type) and issubclass(environment, Environment):
            environment = environment(**kwargs)
            assert isinstance(environment, Environment)
            return Environment2.create(
                environment=environment, max_episode_timesteps=max_episode_timesteps
            )

        elif isinstance(environment, dict):
            # Dictionary specification
            util.deep_disjoint_update(target=kwargs, source=environment)
            environment = kwargs.pop('environment', kwargs.pop('type', 'default'))
            assert environment is not None
            if max_episode_timesteps is None:
                max_episode_timesteps = kwargs.pop('max_episode_timesteps', None)

            return Environment2.create(
                environment=environment, max_episode_timesteps=max_episode_timesteps, **kwargs
            )

        elif isinstance(environment, str):
            if os.path.isfile(environment):
                # JSON file specification
                with open(environment, 'r') as fp:
                    environment = json.load(fp=fp)

                util.deep_disjoint_update(target=kwargs, source=environment)
                environment = kwargs.pop('environment', kwargs.pop('type', 'default'))
                assert environment is not None
                if max_episode_timesteps is None:
                    max_episode_timesteps = kwargs.pop('max_episode_timesteps', None)

                return Environment2.create(
                    environment=environment, max_episode_timesteps=max_episode_timesteps, **kwargs
                )

            elif '.' in environment:
                # Library specification
                library_name, module_name = environment.rsplit('.', 1)
                library = importlib.import_module(name=library_name)
                environment = getattr(library, module_name)
                return Environment2.create(
                    environment=environment, max_episode_timesteps=max_episode_timesteps, **kwargs
                )

            elif environment in tensorforce.environments.environments:
                # Keyword specification
                environment = tensorforce.environments.environments[environment]
                return Environment2.create(
                    environment=environment, max_episode_timesteps=max_episode_timesteps, **kwargs
                )

            else:
                # Default: OpenAI Gym
                try:
                    return Environment2.create(
                        environment='gym', level=environment,
                        max_episode_timesteps=max_episode_timesteps, **kwargs
                    )
                except TensorforceError:
                    raise TensorforceError.value(
                        name='Environment2.create', argument='environment', value=environment
                    )

        else:
            # Default: OpenAI Gym
            from gym import Env
            if isinstance(environment, Env) or \
                    (isinstance(environment, type) and issubclass(environment, Env)):
                return Environment2.create(
                    environment='gym', level=environment,
                    max_episode_timesteps=max_episode_timesteps, **kwargs
                )

            else:
                raise TensorforceError.type(
                    name='Environment2.create', argument='environment', dtype=type(environment)
                )

    def states(self):
        pass

    def actions(self):
        pass

    def reset(self):
        pass

    def execute(self, actions):
        pass


class EnvironmentWrapper2(Environment2):

    def __init__(self, environment, max_episode_timesteps):
        super().__init__()

        if isinstance(environment, EnvironmentWrapper2):
            raise TensorforceError.unexpected()
        if environment.max_episode_timesteps() is not None and \
                max_episode_timesteps is not None and \
                environment.max_episode_timesteps() < max_episode_timesteps:
            raise TensorforceError.unexpected()

        self._environment = environment
        if max_episode_timesteps is None:
            self._max_episode_timesteps = self._environment.max_episode_timesteps()
        else:
            self._max_episode_timesteps = max_episode_timesteps
            if self._environment.max_episode_timesteps() is None:
                self._environment.max_episode_timesteps = (lambda self: max_episode_timesteps)
        self._timestep = None

    def __str__(self):
        return str(self._environment)

    def set_level(self, level: int):
        if isinstance(self._environment, OpenAIGym) and isinstance(self._environment.environment, LilysGardenEnv):
            self._environment.environment.set_level(level)
            self.reset()

    def states(self):
        return self._environment.states()

    def actions(self):
        return self._environment.actions()

    def max_episode_timesteps(self):
        return self._max_episode_timesteps

    def close(self):
        return self._environment.close()

    def reset(self):
        self._timestep = 0
        states = self._environment.reset()
        if isinstance(states, dict):
            states = states.copy()
        return states

    def execute(self, actions):
        if self._timestep is None:
            raise TensorforceError(
                message="An environment episode has to be initialized by calling reset() first."
            )
        assert self._max_episode_timesteps is None or self._timestep < self._max_episode_timesteps
        states, terminal, reward = self._environment.execute(actions=actions)
        if isinstance(states, dict):
            states = states.copy()
        terminal = int(terminal)
        self._timestep += 1
        if terminal == 0 and self._max_episode_timesteps is not None and \
                self._timestep >= self._max_episode_timesteps:
            terminal = 2
        if terminal > 0:
            self._timestep = None
        return states, terminal, reward

    _ATTRIBUTES = frozenset([
        '_actions', 'create', '_environment', '_expect_receive', '_max_episode_timesteps',
        '_timestep'
    ])

    def __getattr__(self, name):
        if name in EnvironmentWrapper2._ATTRIBUTES:
            return super().__getattr__(name)
        else:
            return getattr(self._environment, name)

    def __setattr__(self, name, value):
        if name in EnvironmentWrapper2._ATTRIBUTES:
            super().__setattr__(name, value)
        else:
            return setattr(self._environment, name, value)
