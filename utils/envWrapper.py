import gymnasium as gym

class envWrapper(gym.Wrapper):
    def __getattr__(self, name):
        """
        If this wrapper doesn't have the attribute, 
        forward it to the wrapped environment.
        """
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.env.env.env, name)