from collections import defaultdict

import numpy as np

from .base_sampler import BaseSampler


class SimpleSampler(BaseSampler):
    def __init__(self, **kwargs):
        super(SimpleSampler, self).__init__(**kwargs)

        self._path_length = 0
        self._path_return = 0
        self._current_path = defaultdict(list)
        self._last_path_return = 0
        self._max_path_return = -np.inf
        self._n_episodes = 0
        self._current_observation = None
        self._total_samples = 0

    def _process_observations(self,
                              observation,
                              action,
                              reward,
                              terminal,
                              next_observation,
                              info):
        processed_observation = {
            'observations': observation,
            'actions': action,
            'rewards': [reward],
            'terminals': [terminal],
            'next_observations': next_observation,
            'infos': info,
        }

        return processed_observation

    def sample(self, bmpo=None,candidate_num = 1000):
        if self._current_observation is None:
            self._current_observation = self.env.reset()
        if (bmpo is None or bmpo._planning_horizon == 0):
            action = self.policy.actions_np([
                self.env.convert_to_active_observation(
                    self._current_observation)[None]
            ])[0]
        else:
            # MPC
            s = self.env.convert_to_active_observation(
                self._current_observation)[None]
            states = s.repeat(candidate_num,axis=0)
            actions = bmpo._session.run(bmpo._actions,
                                             feed_dict={
                                                 bmpo._observations_ph: states
                                             })
            next_states, rewards, done, info = bmpo.f_fake_env.step(states, actions)
            rewards = np.squeeze(rewards)
            not_done = ~np.squeeze(done)
            discount = bmpo._discount
            states = next_states
            with self.policy.set_deterministic(True):
                for i in range(bmpo._planning_horizon-1):
                    temp_actions = self.policy.actions_np(states)
                    next_states, temp_rewards, temp_done, info = bmpo.f_fake_env.step(states, temp_actions)
                    rewards += np.squeeze(temp_rewards) * discount * not_done
                    discount *= bmpo._discount
                    not_done *= ~np.squeeze(temp_done)
                    states = next_states

            values = rewards + discount * not_done * np.squeeze(
                bmpo._session.run(bmpo._target_value, feed_dict={
                    bmpo._observations_ph: next_states
                }))
            action = actions[np.argmax(values)]

        next_observation, reward, terminal, info = self.env.step(action)
        self._path_length += 1
        self._path_return += reward
        self._total_samples += 1

        processed_sample = self._process_observations(
            observation=self._current_observation,
            action=action,
            reward=reward,
            terminal=terminal,
            next_observation=next_observation,
            info=info,
        )

        for key, value in processed_sample.items():
            self._current_path[key].append(value)

        if terminal or self._path_length >= self._max_path_length:
            last_path = {
                field_name: np.array(values)
                for field_name, values in self._current_path.items()
            }
            self.pool.add_path(last_path)
            self._last_n_paths.appendleft(last_path)

            self._max_path_return = max(self._max_path_return,
                                        self._path_return)
            self._last_path_return = self._path_return

            self.policy.reset()
            self._current_observation = None
            self._path_length = 0
            self._path_return = 0
            self._current_path = defaultdict(list)

            self._n_episodes += 1
        else:
            self._current_observation = next_observation

        return next_observation, reward, terminal, info

    def random_batch(self, batch_size=None, **kwargs):
        batch_size = batch_size or self._batch_size
        observation_keys = getattr(self.env, 'observation_keys', None)

        return self.pool.random_batch(
            batch_size, observation_keys=observation_keys, **kwargs)

    def get_diagnostics(self):
        diagnostics = super(SimpleSampler, self).get_diagnostics()
        diagnostics.update({
            'max-path-return': self._max_path_return,
            'last-path-return': self._last_path_return,
            'episodes': self._n_episodes,
            'total-samples': self._total_samples,
        })

        return diagnostics