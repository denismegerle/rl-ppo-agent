import abc

# --- EXPERIENCE_BUFFER -------------------------------------------------------
# -----------------------------------------------------------------------------
class ExperienceBuffer(abc.ABC):
  """
  ExperienceBuffer base class, each experience is defined as list of components
  sharing the same time.

  Raises:
    NotImplementedError: since abstract base class
  """

  def __init__(self, num_elem_components : int):
    """
    Constructing an experience buffer with a specified amount of components.

    Args:
        num_elem_components (int): amount of components per timestep
    """
    self.num_elem_components = num_elem_components

  @abc.abstractmethod
  def add_element(self, element : list):
    """
    Adding an element (one timestep).

    Args:
        element (list): list of components

    Raises:
        NotImplementedError: since abstract base class
    """
    raise NotImplementedError('not implemented yet')

  @abc.abstractmethod
  def add_batch(self, batch : list):
    """
    Adding a batch of elements (n = len(batch) timesteps), where each element
    of the list corresponds to n timesteps of a component

    Args:
        batch (list): list of components, each containing n timesteps

    Raises:
        NotImplementedError: since abstract base class
    """
    raise NotImplementedError('not implemented yet')

  @abc.abstractmethod
  def sample(self):
    """
    Sampling from this experience buffer in the form of
      (a1, a2, ..., at), ...

    Raises:
        NotImplementedError: since abstract base class
    """
    raise NotImplementedError('not implemented yet')
  
  def reset(self):
    """
    Resetting (emptying) interal buffer completely.
    """
    self.__init__(self.num_elem_components)


# --- CONSECUTIVE_EXPERIENCE_BUFFER -------------------------------------------
# -----------------------------------------------------------------------------
class ConsecutiveExperienceBuffer(ExperienceBuffer):
  """
  Consecutive Buffer, where newly added element should be the successor of
  the last element in the buffer. Also, the buffer is layed out componentwise,
  receiving experiences (elements) in the form of
    (a1, b1, ...), ..., (at, bt, ...)           [I]
    [where a, b, ... are components for each timestep]
  and maintaining them such that each component is clustered together like so
    (a1, a2, ..., at), (b1, b2, ...), ...       [II]

  Args:
      ExperienceBuffer ([type]): base class experience buffer
  """
  
  def __init__(self, num_elem_components : int):
    """
    see baseclass
    """
    super().__init__(num_elem_components)
    self.buffer = [[] for i in range(self.num_elem_components)]

  def add_element(self, element : list):
    """
    Adds an element to the buffer depthwise.

    Args:
        element (list): list of components sharing the same time step
    """
    assert len(element) == self.num_elem_components, 'element does not contain {self.num_elem_components} states'
    for stx in range(self.num_elem_components):
      self.buffer[stx].append(element[stx])
  
  def add_batch(self, batch : list):
    """
    Adds a batch of elements to the buffer. Note that the form is already as defined in [II].

    Args:
        batch (list): list of components, with each component containing n timesteps.
    """
    assert len(batch) == self.num_elem_components, 'batch elements do not contain {self.num_elem_components} states'
    for stx in range(self.num_elem_components):
      self.buffer[stx] += batch[stx]

  def sample(self):
    """
    Sampling a consecutive buffer yields all time steps of all components directly.

    Returns:
        list: list of components, each component containing n time steps
    """
    return self.buffer