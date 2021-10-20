import numpy as np
import tensorflow as tf


def assert_normalized(vector,
                      order='euclidean',
                      axis=-1,
                      eps=None,
                      name='assert_normalized'):
  """Checks whether vector/quaternion is normalized in its last dimension.
  Note:
    In the following, A1 to An are optional batch dimensions.
  Args:
    vector: A tensor of shape `[A1, ..., M, ..., An]`, where the axis of M
      contains the vectors.
    order: Order of the norm passed to tf.norm.
    axis: The axis containing the vectors.
    eps: A `float` describing the tolerance used to determine if the norm is
      equal to `1.0`.
    name: A name for this op. Defaults to 'assert_normalized'.
  Raises:
    InvalidArgumentError: If the norm of `vector` is not `1.0`.
  Returns:
    The input vector, with dependence on the assertion operator in the graph.
  """


  with tf.name_scope(name):
    vector = tf.convert_to_tensor(value=vector)
    if eps is None:
      eps = select_eps_for_division(vector.dtype)
    eps = tf.convert_to_tensor(value=eps, dtype=vector.dtype)

    norm = tf.norm(tensor=vector, ord=order, axis=axis)
    one = tf.constant(1.0, dtype=norm.dtype)
    with tf.control_dependencies(
        [tf.debugging.assert_near(norm, one, atol=eps)]):
      return tf.identity(vector)

def from_axis_angle(axis, angle, name="rotation_matrix_3d_from_axis_angle"):
  """Convert an axis-angle representation to a rotation matrix.
  Note:
    In the following, A1 to An are optional batch dimensions, which must be
    broadcast compatible.
  Args:
    axis: A tensor of shape `[A1, ..., An, 3]`, where the last dimension
      represents a normalized axis.
    angle: A tensor of shape `[A1, ..., An, 1]`, where the last dimension
      represents a normalized axis.
    name: A name for this op that defaults to
      "rotation_matrix_3d_from_axis_angle".
  Returns:
    A tensor of shape `[A1, ..., An, 3, 3]`, where the last two dimensions
    represents a 3d rotation matrix.
  Raises:
    ValueError: If the shape of `axis` or `angle` is not supported.
  """
  with tf.name_scope(name):
    axis = tf.convert_to_tensor(value=axis)
    angle = tf.convert_to_tensor(value=angle)

    # shape.check_static(tensor=axis, tensor_name="axis", has_dim_equals=(-1, 3))
    # shape.check_static(
    #     tensor=angle, tensor_name="angle", has_dim_equals=(-1, 1))
    # shape.compare_batch_dimensions(
    #     tensors=(axis, angle),
    #     tensor_names=("axis", "angle"),
    #     last_axes=-2,
    #     broadcast_compatible=True)
    axis = assert_normalized(axis)

    sin_axis = tf.sin(angle) * axis
    cos_angle = tf.cos(angle)
    cos1_axis = (1.0 - cos_angle) * axis
    _, axis_y, axis_z = tf.unstack(axis, axis=-1)
    cos1_axis_x, cos1_axis_y, _ = tf.unstack(cos1_axis, axis=-1)
    sin_axis_x, sin_axis_y, sin_axis_z = tf.unstack(sin_axis, axis=-1)
    tmp = cos1_axis_x * axis_y
    m01 = tmp - sin_axis_z
    m10 = tmp + sin_axis_z
    tmp = cos1_axis_x * axis_z
    m02 = tmp + sin_axis_y
    m20 = tmp - sin_axis_y
    tmp = cos1_axis_y * axis_z
    m12 = tmp - sin_axis_x
    m21 = tmp + sin_axis_x
    diag = cos1_axis * axis + cos_angle
    diag_x, diag_y, diag_z = tf.unstack(diag, axis=-1)
    matrix = tf.stack((diag_x, m01, m02,
                       m10, diag_y, m12,
                       m20, m21, diag_z),
                      axis=-1)  # pyformat: disable
    output_shape = tf.concat((tf.shape(input=axis)[:-1], (3, 3)), axis=-1)
    return tf.reshape(matrix, shape=output_shape)

def select_eps_for_division(dtype):
  """Selects default values for epsilon to make divisions safe based on dtype.
  This function returns an epsilon slightly greater than the smallest positive
  floating number that is representable for the given dtype. This is mainly used
  to prevent division by zero, which produces Inf values. However, if the
  nominator is orders of magnitude greater than `1.0`, eps should also be
  increased accordingly. Only floating types are supported.
  Args:
    dtype: The `tf.DType` of the tensor to which eps will be added.
  Raises:
    ValueError: If `dtype` is not a floating type.
  Returns:
    A `float` to be used to make operations safe.
  """
  return 10.0 * np.finfo(dtype.as_numpy_dtype).tiny